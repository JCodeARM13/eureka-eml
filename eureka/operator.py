"""EML operator: eml(x, y) = exp(x) - ln(y)."""

import torch


def eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.exp(x) - torch.log(torch.clamp(y, min=1e-10))


def safe_eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_safe = torch.clamp(x, -20.0, 20.0)
    y_safe = torch.clamp(y, min=1e-10)
    result = torch.exp(x_safe) - torch.log(y_safe)
    result = torch.clamp(result, -1e6, 1e6)
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))
