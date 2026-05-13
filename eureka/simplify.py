"""Convert EML trees to human-readable equations using SymPy."""

import math
import sympy
from sympy import (
    exp, log, Symbol, Integer, Float, simplify, nsimplify,
    latex, count_ops, Number, pi, E, trigsimp, powsimp, radsimp,
)

_SNAP_CONSTANTS = [
    (0, Integer(0)),
    (1, Integer(1)),
    (2, Integer(2)),
    (3, Integer(3)),
    (math.e, E),
    (math.pi, pi),
]


def tree_to_sympy(tree) -> sympy.Expr:
    name = type(tree).__name__
    if name == "ConstantLeaf":
        return Integer(1)
    elif name == "VariableLeaf":
        return Symbol(f"x{tree.index}")
    elif name == "ParameterLeaf":
        v = tree.value.item()
        for target, sym_val in _SNAP_CONSTANTS:
            if abs(v - target) < 0.05:
                return sym_val
        return Float(round(v, 2))
    elif name == "EMLNode":
        left_expr = tree_to_sympy(tree.left)
        right_expr = tree_to_sympy(tree.right)
        raw = exp(left_expr) - log(right_expr)
        try:
            return simplify(raw)
        except Exception:
            return raw
    raise ValueError(f"Unknown node type: {name}")


def simplify_tree(tree) -> str:
    try:
        expr = tree_to_sympy(tree)
        expr = nsimplify(expr, tolerance=0.01)
        for fn in (trigsimp, powsimp, radsimp):
            try:
                expr = fn(expr)
            except Exception:
                pass
        return str(expr)
    except Exception:
        return "Could not simplify"


def to_latex(tree) -> str:
    try:
        expr = tree_to_sympy(tree)
        expr = nsimplify(expr, tolerance=0.01)
        return latex(expr)
    except Exception:
        return r"\text{Could not simplify}"


def simplify_with_denorm(tree, y_mean: float, y_std: float,
                         x_means=None, x_stds=None) -> str:
    try:
        expr = tree_to_sympy(tree)
        denorm = y_std * expr + y_mean
        if x_means is not None and x_stds is not None:
            for i, (xm, xs) in enumerate(zip(x_means, x_stds)):
                xi = Symbol(f"x{i}")
                denorm = denorm.subs(xi, (xi - xm) / xs)
        denorm = simplify(denorm)
        denorm = nsimplify(denorm, tolerance=0.01)
        return str(denorm)
    except Exception:
        return simplify_tree(tree)


def format_equation(tree, target_name: str = "y", feature_names: list[str] | None = None) -> str:
    try:
        expr = tree_to_sympy(tree)
        expr = nsimplify(expr, tolerance=0.01)
        if feature_names:
            for i, name in enumerate(feature_names):
                expr = expr.subs(Symbol(f"x{i}"), Symbol(name))
        return f"{target_name} = {expr}"
    except Exception:
        return f"{target_name} = [could not simplify]"


def estimate_complexity(expr) -> int:
    if isinstance(expr, str):
        expr = sympy.sympify(expr)
    return count_ops(expr) + len(expr.atoms(Symbol, Number))
