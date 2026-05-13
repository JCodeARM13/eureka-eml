"""Benchmark: Test discovery on known equations."""

import numpy as np
import time
from eureka.regression import discover
from eureka.simplify import simplify_tree

BENCHMARKS = [
    ("exp(x)",      lambda x: np.exp(x),           (-2, 2)),
    ("sin(x)",      lambda x: np.sin(x),           (-3, 3)),
    ("x^2",         lambda x: x**2,                (-3, 3)),
    ("ln(x)",       lambda x: np.log(x),           (0.1, 5)),
    ("x*exp(-x)",   lambda x: x * np.exp(-x),      (0, 5)),
]


def run_benchmarks():
    print("Eureka-EML Benchmark Suite")
    print("=" * 70)
    print(f"{'Function':<15} {'MSE':>12} {'Nodes':>6} {'Time':>7} {'Discovered'}")
    print("-" * 70)

    for name, fn, (lo, hi) in BENCHMARKS:
        x = np.linspace(lo, hi, 150).astype(np.float32)
        y = fn(x).astype(np.float32)

        t0 = time.time()
        result = discover(x, y, population=30, generations=50, timeout=45, verbose=False)
        elapsed = time.time() - t0

        eq = simplify_tree(result.tree)
        eq_short = eq[:35] + "..." if len(eq) > 35 else eq
        print(f"{name:<15} {result.mse:>12.4e} {result.complexity:>6} {elapsed:>6.1f}s {eq_short}")

    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
