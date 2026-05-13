"""Demo: Discover physics equations from experimental data."""

import numpy as np
from eureka.regression import discover
from eureka.simplify import simplify_tree, to_latex


def demo_exponential_decay():
    """y = 5 * exp(-0.3 * t)"""
    print("=== Exponential Decay: y = 5 * exp(-0.3t) ===")
    np.random.seed(42)
    t = np.linspace(0, 10, 150).astype(np.float32)
    y = (5.0 * np.exp(-0.3 * t) + np.random.randn(150).astype(np.float32) * 0.1).astype(np.float32)

    result = discover(t, y, population=30, generations=50, timeout=60, verbose=True)
    print(f"\nDiscovered: y = {simplify_tree(result.tree)}")
    print(f"True:       y = 5 * exp(-0.3t)")
    print(f"MSE: {result.mse:.6e}\n")


def demo_ohms_law():
    """V = I * R"""
    print("=== Ohm's Law: V = I * R ===")
    np.random.seed(42)
    n = 150
    I = np.random.uniform(0.1, 10, n).astype(np.float32)
    R = np.random.uniform(1, 100, n).astype(np.float32)
    V = (I * R + np.random.randn(n).astype(np.float32) * 0.5).astype(np.float32)

    X = np.column_stack([I, R])
    result = discover(X, V, population=30, generations=50, timeout=60, verbose=True)
    print(f"\nDiscovered: V = {simplify_tree(result.tree)}")
    print(f"True:       V = I * R")
    print(f"MSE: {result.mse:.6e}\n")


if __name__ == "__main__":
    print("Eureka-EML Physics Demos")
    print("=" * 50 + "\n")
    demo_exponential_decay()
