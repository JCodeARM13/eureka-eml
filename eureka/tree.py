"""EML tree representation as differentiable PyTorch modules."""

import random

import torch
import torch.nn as nn

from eureka.operator import safe_eml


class ConstantLeaf(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)


class VariableLeaf(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.index]


class ParameterLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand(x.shape[0])


class EMLNode(nn.Module):
    def __init__(self, left: nn.Module, right: nn.Module):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_out = self.left(x)
        right_out = self.right(x)
        result = safe_eml(left_out, right_out)
        return torch.clamp(result, -1e6, 1e6)


def build_random_tree(depth: int, num_variables: int, p_param: float = 0.3) -> nn.Module:
    if depth == 0:
        r = random.random()
        if r < p_param:
            return ParameterLeaf()
        elif r < p_param + (1 - p_param) / 2:
            return ConstantLeaf()
        else:
            idx = random.randint(0, max(0, num_variables - 1))
            return VariableLeaf(idx)
    left = build_random_tree(depth - 1, num_variables, p_param)
    right = build_random_tree(depth - 1, num_variables, p_param)
    return EMLNode(left, right)


def count_nodes(tree: nn.Module) -> int:
    if isinstance(tree, EMLNode):
        return 1 + count_nodes(tree.left) + count_nodes(tree.right)
    return 1


def tree_depth(tree: nn.Module) -> int:
    if isinstance(tree, EMLNode):
        return 1 + max(tree_depth(tree.left), tree_depth(tree.right))
    return 0
