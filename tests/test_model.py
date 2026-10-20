"""Unit tests for SRCNN model architecture and dimension verification."""

import pytest
import torch
from underwater_sr.model import SRCNN, build_model


def test_srcnn_x2_dimensions():
    """Verify output spatial dimensions match 2x upscaling."""
    model = SRCNN(scale_factor=2)
    model.eval()

    # Synthetic RGB input: Batch=2, Channels=3, H=32, W=32
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 64, 64), f"Expected (2, 3, 64, 64), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output values must be clamped to [0, 1]"


def test_srcnn_x4_dimensions():
    """Verify output spatial dimensions match 4x upscaling."""
    model = SRCNN(scale_factor=4)
    model.eval()

    # Synthetic RGB input: Batch=1, Channels=3, H=24, W=30 (rectangular input)
    x = torch.rand(1, 3, 24, 30)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 96, 120), f"Expected (1, 3, 96, 120), got {out.shape}"


def test_srcnn_invalid_scale():
    """Verify invalid scale factors raise ValueError."""
    with pytest.raises(ValueError):
        SRCNN(scale_factor=3)


def test_cuda_compatibility():
    """Verify CUDA support if a GPU is available, else verify CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(scale_factor=2, device=device)
    x = torch.rand(1, 3, 16, 16, device=device)

    with torch.no_grad():
        out = model(x)

    assert out.device == x.device
    assert out.shape == (1, 3, 32, 32)

