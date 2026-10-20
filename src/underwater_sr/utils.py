"""Utility functions for device management, image conversion, and configs."""

import os
import yaml
import torch
import numpy as np
from PIL import Image


def get_device(preference: str = "auto") -> torch.device:
    """Return torch.device based on user preference and availability."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PyTorch image tensor (C, H, W in [0, 1]) to a PIL Image."""
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    np_img = tensor.permute(1, 2, 0).numpy()
    np_img = (np_img * 255.0).round().astype(np.uint8)
    return Image.fromarray(np_img)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch float tensor (C, H, W in [0, 1])."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    np_img = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    return tensor

