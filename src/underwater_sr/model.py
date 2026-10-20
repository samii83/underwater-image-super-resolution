"""SRCNN-style Super-Resolution model for underwater imagery."""

from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """Lightweight Super-Resolution Convolutional Neural Network (SRCNN).

    Processes RGB images with pre-upsampling via bicubic interpolation, followed
    by patch extraction, non-linear feature mapping, and high-resolution reconstruction.

    Supports arbitrary input sizes, with explicit support for 2x and 4x upscaling.
    Runs on both CPU and CUDA devices.

    Args:
        scale_factor: Upscaling factor (commonly 2 or 4).
        in_channels: Number of input color channels (default: 3 for RGB).
        num_channels: Intermediate representation feature channels (default: 64).
        mid_channels: Non-linear mapping channels (default: 32).
    """

    def __init__(
        self,
        scale_factor: int = 4,
        in_channels: int = 3,
        num_channels: int = 64,
        mid_channels: int = 32,
    ) -> None:
        super().__init__()
        if scale_factor not in (2, 4):
            raise ValueError(f"scale_factor must be 2 or 4, got {scale_factor}")

        self.scale_factor = scale_factor
        self.in_channels = in_channels

        # Layer 1: Patch extraction and representation (9x9 conv)
        self.conv1 = nn.Conv2d(
            in_channels, num_channels, kernel_size=9, padding=4, padding_mode="replicate"
        )
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 2: Non-linear mapping (5x5 conv)
        self.conv2 = nn.Conv2d(
            num_channels, mid_channels, kernel_size=5, padding=2, padding_mode="replicate"
        )
        self.relu2 = nn.ReLU(inplace=True)

        # Layer 3: Reconstruction (5x5 conv)
        self.conv3 = nn.Conv2d(
            mid_channels, in_channels, kernel_size=5, padding=2, padding_mode="replicate"
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with He / Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input low-resolution tensor of shape (B, C, H, W) in [0, 1].

        Returns:
            Super-resolved tensor of shape (B, C, H * scale_factor, W * scale_factor).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got tensor of shape {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")

        # Upscale low-resolution input using bicubic interpolation
        x_upscaled = F.interpolate(
            x,
            scale_factor=float(self.scale_factor),
            mode="bicubic",
            align_corners=False,
        )

        # Feature extraction -> Non-linear mapping -> Reconstruction
        out = self.relu1(self.conv1(x_upscaled))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)

        # Residual connection to facilitate convergence on underwater images
        out = out + x_upscaled
        return torch.clamp(out, 0.0, 1.0)


def build_model(
    scale_factor: int = 4,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> SRCNN:
    """Helper factory function to instantiate and place model onto device."""
    model = SRCNN(scale_factor=scale_factor, **kwargs)
    if isinstance(device, str):
        device = torch.device(device)
    return model.to(device)

