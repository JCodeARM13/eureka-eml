"""Quick start demo: discover y = x^2 from data."""

import numpy as np
from eureka.regression import discover
from eureka.simplify import simplify_tree, to_latex

x = np.linspace(-5, 5, 200).astype(np.float32)
y = (x ** 2).astype(np.float32)
X = x.reshape(-1, 1)

print("Eureka-EML Quick Start")
print("=" * 40)
print(f"Data: {len(x)} points of y = x^2")
print()

result = discover(X, y, population=30, generations=50, timeout=60, verbose=True)

equation = simplify_tree(result.tree)
print()
print("=" * 40)
print(f"Discovered: y = {equation}")
print(f"LaTeX:      y = {to_latex(result.tree)}")
print(f"MSE: {result.mse:.6e}")
print(f"Complexity: {result.complexity} nodes")
