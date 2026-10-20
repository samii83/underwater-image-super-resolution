"""Image quality metrics for super-resolution evaluation (PSNR and SSIM)."""

import math
import torch
import torch.nn.functional as F
import numpy as np


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        pred: Predicted tensor (B, C, H, W) or (C, H, W) in [0, 1].
        target: Ground truth tensor of the same shape in [0, 1].
        max_val: Dynamic range of pixel values (default: 1.0).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Mean PSNR value across batch as a Python float.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 100.0
    return float(10.0 * math.log10((max_val ** 2) / mse))


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_ssim_window(window_size: int, channel: int) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM."""
    _1d = _gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Args:
        pred: Predicted tensor (B, C, H, W) or (C, H, W) in [0, 1].
        target: Ground truth tensor of same shape in [0, 1].
        window_size: Gaussian window dimension (default: 11).
        max_val: Maximum pixel value range (default: 1.0).

    Returns:
        SSIM value as a Python float in [-1, 1].
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    channel = pred.size(1)
    window = _create_ssim_window(window_size, channel).to(pred.device, dtype=pred.dtype)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())

