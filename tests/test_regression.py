"""Tests for the regression engine."""

import numpy as np
import torch
from eureka.regression import discover, optimize_tree, DiscoveryResult
from eureka.tree import build_random_tree


def test_optimize_tree():
    tree = build_random_tree(depth=2, num_variables=1, p_param=0.5)
    X = torch.randn(50, 1)
    y = torch.randn(50)
    mse = optimize_tree(tree, X, y, lr=0.01, steps=100)
    assert isinstance(mse, float)
    assert mse >= 0


def test_discover_returns_result():
    X = np.linspace(-2, 2, 50).reshape(-1, 1).astype(np.float32)
    y = (X[:, 0] ** 2).astype(np.float32)
    result = discover(X, y, population=10, generations=5, timeout=30, verbose=False)
    assert isinstance(result, DiscoveryResult)
    assert result.mse >= 0
    assert result.complexity > 0
    assert result.tree is not None


def test_discover_1d_input():
    x = np.linspace(0, 5, 30).astype(np.float32)
    y = np.exp(x * 0.5).astype(np.float32)
    result = discover(x, y, population=10, generations=5, timeout=30, verbose=False)
    assert isinstance(result, DiscoveryResult)


def test_discover_timeout():
    X = np.random.randn(100, 3).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    result = discover(X, y, population=10, generations=1000, timeout=5, verbose=False)
    assert isinstance(result, DiscoveryResult)
