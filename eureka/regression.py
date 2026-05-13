"""Evolutionary + gradient hybrid symbolic regression engine."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from eureka.tree import (
    EMLNode,
    ConstantLeaf,
    VariableLeaf,
    ParameterLeaf,
    build_random_tree,
    count_nodes,
    tree_depth,
)


@dataclass
class DiscoveryResult:
    formula: str
    sympy_expr: object = None
    mse: float = float("inf")
    complexity: int = 0
    tree: nn.Module = field(default=None, repr=False)
    y_mean: float = 0.0
    y_std: float = 1.0


def _to_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float()
    return arr.float()


def _normalize(X: torch.Tensor, y: torch.Tensor):
    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0) + 1e-8
    y_mean = y.mean().item()
    y_std = y.std().item() + 1e-8
    X_n = (X - x_mean) / x_std
    y_n = (y - y_mean) / y_std
    return X_n, y_n, y_mean, y_std


def _collect_leaves(module: nn.Module) -> list[tuple[nn.Module, str, nn.Module]]:
    leaves = []
    for name, child in module.named_children():
        if isinstance(child, (ConstantLeaf, VariableLeaf, ParameterLeaf)):
            leaves.append((module, name, child))
        elif isinstance(child, EMLNode):
            leaves.extend(_collect_leaves(child))
    return leaves


def _collect_internal(module: nn.Module) -> list[tuple[nn.Module, str, EMLNode]]:
    internals = []
    for name, child in module.named_children():
        if isinstance(child, EMLNode):
            internals.append((module, name, child))
            internals.extend(_collect_internal(child))
    return internals


def _random_leaf(num_variables: int) -> nn.Module:
    kind = random.choice(["constant", "variable", "parameter"])
    if kind == "constant":
        return ConstantLeaf()
    elif kind == "variable":
        return VariableLeaf(random.randint(0, max(0, num_variables - 1)))
    return ParameterLeaf()


def optimize_tree(tree: nn.Module, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.05, steps: int = 1000) -> float:
    params = [p for p in tree.parameters() if p.requires_grad]
    if not params:
        with torch.no_grad():
            pred = tree(X)
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                return 1e10
            return torch.mean((pred - y) ** 2).item()

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5
    )
    best_mse = 1e10
    stale = 0

    for _ in range(steps):
        optimizer.zero_grad()
        pred = tree(X)

        if torch.isnan(pred).any() or torch.isinf(pred).any():
            return best_mse if best_mse < 1e10 else 1e10

        loss = torch.mean((pred - y) ** 2)
        mse_val = loss.item()

        if not np.isfinite(mse_val):
            return best_mse if best_mse < 1e10 else 1e10

        if mse_val < best_mse - 1e-8:
            best_mse = mse_val
            stale = 0
        else:
            stale += 1

        if best_mse < 1e-10 or stale > 100:
            return best_mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        scheduler.step(mse_val)

    return best_mse


def mutate_tree(tree: nn.Module, num_variables: int) -> nn.Module:
    new_tree = copy.deepcopy(tree)
    leaves = _collect_leaves(new_tree)
    internals = _collect_internal(new_tree)

    mutations = ["swap_leaf"]
    if leaves:
        mutations.append("grow")
    if internals:
        mutations.append("prune")

    action = random.choice(mutations)

    if action == "grow" and leaves:
        parent, attr, _ = random.choice(leaves)
        subtree = build_random_tree(random.randint(1, 2), num_variables, p_param=0.3)
        setattr(parent, attr, subtree)
    elif action == "prune" and internals:
        parent, attr, _ = random.choice(internals)
        setattr(parent, attr, _random_leaf(num_variables))
    elif action == "swap_leaf" and leaves:
        parent, attr, old_leaf = random.choice(leaves)
        for _ in range(5):
            new_leaf = _random_leaf(num_variables)
            if type(new_leaf) is not type(old_leaf):
                break
        setattr(parent, attr, new_leaf)

    return new_tree


def crossover(tree_a: nn.Module, tree_b: nn.Module, num_variables: int) -> nn.Module:
    child = copy.deepcopy(tree_a)
    internals_child = _collect_internal(child)
    internals_donor = _collect_internal(tree_b)

    if not internals_child or not internals_donor:
        return mutate_tree(child, num_variables)

    parent, attr, _ = random.choice(internals_child)
    _, _, donor_subtree = random.choice(internals_donor)
    setattr(parent, attr, copy.deepcopy(donor_subtree))
    return child


def _build_diverse_population(size: int, max_depth: int, num_variables: int) -> list[nn.Module]:
    pop = []
    n_shallow = int(size * 0.3)
    n_medium = int(size * 0.4)
    n_deep = size - n_shallow - n_medium

    for _ in range(n_shallow):
        pop.append(build_random_tree(random.randint(1, 2), num_variables, p_param=0.3))
    for _ in range(n_medium):
        pop.append(build_random_tree(random.randint(3, min(4, max_depth)), num_variables, p_param=0.3))
    for _ in range(n_deep):
        pop.append(build_random_tree(random.randint(min(5, max_depth), max_depth), num_variables, p_param=0.3))

    return pop


def discover(
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    population: int = 30,
    generations: int = 50,
    max_depth: int = 6,
    timeout: float = 120,
    restarts: int = 3,
    verbose: bool = True,
) -> DiscoveryResult:
    X_t = _to_tensor(X)
    y_t = _to_tensor(y)

    if X_t.ndim == 1:
        X_t = X_t.unsqueeze(1)

    num_variables = X_t.shape[1]

    global_best_tree = None
    global_best_mse = float("inf")
    global_best_complexity = 0
    total_start = time.time()
    restart_timeout = timeout / restarts

    for restart in range(restarts):
        if time.time() - total_start > timeout:
            break

        restart_start = time.time()
        pop = _build_diverse_population(population, max_depth, num_variables)
        best_tree = None
        best_mse = float("inf")
        best_complexity = 0

        for gen in range(generations):
            if time.time() - restart_start > restart_timeout:
                if verbose:
                    print(f"  Restart {restart+1}: timeout at gen {gen}")
                break

            scores: list[tuple[float, int, nn.Module]] = []
            for tree in pop:
                opt_steps = 150 if gen < 15 else 300 if gen < 30 else 500
                mse = optimize_tree(tree, X_t, y_t, lr=0.05, steps=opt_steps)
                nodes = count_nodes(tree)
                fitness = mse + 0.005 * nodes
                scores.append((fitness, nodes, tree))

                if mse < best_mse:
                    best_mse = mse
                    best_tree = copy.deepcopy(tree)
                    best_complexity = nodes

            scores.sort(key=lambda t: t[0])

            if verbose and gen % 10 == 0:
                print(f"  R{restart+1} Gen {gen:>3d} | MSE: {best_mse:.6e} | Nodes: {best_complexity}")

            if best_mse < 1e-8:
                if verbose:
                    print(f"  Restart {restart+1}: converged at gen {gen}")
                break

            keep = max(2, int(len(scores) * 0.4))
            survivors = [tree for _, _, tree in scores[:keep]]
            new_pop = [copy.deepcopy(t) for t in survivors]

            slots = population - len(new_pop) - 3
            n_crossover = max(1, int(slots * 0.2))

            for _ in range(n_crossover):
                a, b = random.sample(survivors, min(2, len(survivors)))
                child = crossover(a, b, num_variables)
                if tree_depth(child) <= max_depth:
                    new_pop.append(child)

            while len(new_pop) < population - 3:
                parent = random.choice(survivors)
                child = mutate_tree(parent, num_variables)
                if tree_depth(child) <= max_depth:
                    new_pop.append(child)
                else:
                    new_pop.append(copy.deepcopy(parent))

            for _ in range(min(3, population - len(new_pop))):
                d = random.randint(2, min(5, max_depth))
                new_pop.append(build_random_tree(d, num_variables, p_param=0.3))

            pop = new_pop

        if best_mse < global_best_mse:
            global_best_mse = best_mse
            global_best_tree = best_tree
            global_best_complexity = best_complexity

    if global_best_tree is None:
        global_best_tree = build_random_tree(2, num_variables, p_param=0.3)
        global_best_mse = 1e10
        global_best_complexity = count_nodes(global_best_tree)

    if verbose:
        print(f"Best across {restarts} restarts: MSE={global_best_mse:.6e}")

    return DiscoveryResult(
        formula="EML tree (run simplify for readable form)",
        sympy_expr=None,
        mse=global_best_mse,
        complexity=global_best_complexity,
        tree=global_best_tree,
    )
