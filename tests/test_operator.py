"""Tests for the EML operator."""

import torch
from eureka.operator import eml, safe_eml


def test_eml_basic():
    x = torch.tensor([1.0])
    y = torch.tensor([1.0])
    result = eml(x, y)
    expected = torch.exp(x) - torch.log(y)
    assert torch.allclose(result, expected)


def test_eml_exp_shortcut():
    x = torch.tensor([2.0])
    y = torch.tensor([1.0])
    result = eml(x, y)
    assert torch.allclose(result, torch.exp(x), atol=1e-6)


def test_safe_eml_handles_negative_y():
    x = torch.tensor([1.0])
    y = torch.tensor([-5.0])
    result = safe_eml(x, y)
    assert torch.isfinite(result).all()


def test_safe_eml_handles_large_x():
    x = torch.tensor([1000.0])
    y = torch.tensor([1.0])
    result = safe_eml(x, y)
    assert torch.isfinite(result).all()
    assert result.abs().item() <= 1e6


def test_eml_batch():
    x = torch.randn(100)
    y = torch.abs(torch.randn(100)) + 0.1
    result = safe_eml(x, y)
    assert result.shape == (100,)
    assert torch.isfinite(result).all()
