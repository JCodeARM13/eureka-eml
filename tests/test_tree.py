"""Tests for the EML tree modules."""

import torch
from eureka.tree import (
    ConstantLeaf, VariableLeaf, ParameterLeaf, EMLNode,
    build_random_tree, count_nodes, tree_depth,
)


def test_constant_leaf():
    leaf = ConstantLeaf()
    X = torch.randn(10, 3)
    out = leaf(X)
    assert out.shape == (10,)
    assert torch.allclose(out, torch.ones(10))


def test_variable_leaf():
    leaf = VariableLeaf(1)
    X = torch.randn(10, 3)
    out = leaf(X)
    assert torch.allclose(out, X[:, 1])


def test_parameter_leaf_gradient():
    leaf = ParameterLeaf()
    X = torch.randn(10, 2)
    out = leaf(X)
    loss = out.sum()
    loss.backward()
    assert leaf.value.grad is not None


def test_eml_node_forward():
    left = ConstantLeaf()
    right = ConstantLeaf()
    node = EMLNode(left, right)
    X = torch.randn(10, 2)
    out = node(X)
    assert out.shape == (10,)
    assert torch.isfinite(out).all()


def test_build_random_tree():
    tree = build_random_tree(depth=3, num_variables=2)
    assert isinstance(tree, EMLNode)
    X = torch.randn(20, 2)
    out = tree(X)
    assert out.shape == (20,)


def test_count_nodes():
    tree = build_random_tree(depth=2, num_variables=2)
    n = count_nodes(tree)
    assert n >= 3


def test_tree_depth_matches():
    tree = build_random_tree(depth=3, num_variables=2)
    d = tree_depth(tree)
    assert d <= 3


def test_gradients_flow():
    tree = build_random_tree(depth=3, num_variables=2, p_param=0.5)
    X = torch.randn(50, 2)
    y = torch.randn(50)
    pred = tree(X)
    loss = (pred - y).pow(2).mean()
    loss.backward()
