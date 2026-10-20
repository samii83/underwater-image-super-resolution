"""Unit tests for PSNR and SSIM metric calculation."""

import torch
from underwater_sr.metrics import calculate_psnr, calculate_ssim


def test_metrics_identical_images():
    """Verify metrics on identical images produce expected peak scores."""
    img = torch.rand(1, 3, 64, 64)

    psnr = calculate_psnr(img, img)
    assert psnr >= 90.0, f"Expected high PSNR for identical images, got {psnr}"

    ssim = calculate_ssim(img, img)
    assert abs(ssim - 1.0) < 1e-4, f"Expected SSIM ~1.0, got {ssim}"


def test_metrics_perturbed_images():
    """Verify added noise decreases PSNR and SSIM."""
    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.clamp(img1 + torch.randn_like(img1) * 0.1, 0.0, 1.0)

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)

    assert 10.0 < psnr < 40.0
    assert 0.0 < ssim < 1.0

