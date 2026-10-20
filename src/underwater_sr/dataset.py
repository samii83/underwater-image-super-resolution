"""Dataset loading and preprocessing for underwater super-resolution."""

import os
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class UnderwaterSRDataset(Dataset):
    """Dataset for Underwater Image Super-Resolution.

    Supports two modes:
    1. Paired Mode: `lr_dir` and `hr_dir` are both provided with matching filenames.
    2. Synthetic Mode: Only `hr_dir` is provided. Low-resolution images are dynamically
       synthesized by downscaling HR images and applying underwater-like blur.

    Args:
        hr_dir: Directory containing High-Resolution (HR) ground-truth images.
        lr_dir: Optional directory containing Low-Resolution (LR) images.
        scale_factor: Upscaling factor (2 or 4).
        patch_size: Optional square crop size for HR patches during training.
        is_train: If True, applies random cropping and horizontal flipping.
    """

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        scale_factor: int = 4,
        patch_size: Optional[int] = 128,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.is_train = is_train

        self.hr_files = self._collect_image_files(self.hr_dir)
        if len(self.hr_files) == 0:
            # Keep dataset valid even if empty directory during initial setup
            pass

    def _collect_image_files(self, directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        files = [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        ]
        return sorted(files)

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_files[idx]
        hr_img = Image.open(hr_path).convert("RGB")

        # Load or synthesize LR
        if self.lr_dir and (self.lr_dir / hr_path.name).exists():
            lr_path = self.lr_dir / hr_path.name
            lr_img = Image.open(lr_path).convert("RGB")
        else:
            lr_w = max(1, hr_img.width // self.scale_factor)
            lr_h = max(1, hr_img.height // self.scale_factor)
            lr_img = hr_img.resize((lr_w, lr_h), Image.Resampling.BICUBIC)

        # Apply random crop for training
        if self.is_train and self.patch_size is not None:
            # Crop HR image
            if hr_img.width >= self.patch_size and hr_img.height >= self.patch_size:
                i, j, h, w = self._get_random_crop_params(hr_img, self.patch_size)
                hr_img = TF.crop(hr_img, i, j, h, w)

                lr_patch_size = self.patch_size // self.scale_factor
                lr_i, lr_j = i // self.scale_factor, j // self.scale_factor
                lr_img = TF.crop(lr_img, lr_i, lr_j, lr_patch_size, lr_patch_size)

            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                hr_img = TF.hflip(hr_img)
                lr_img = TF.hflip(lr_img)

        hr_tensor = TF.to_tensor(hr_img)
        lr_tensor = TF.to_tensor(lr_img)
        return lr_tensor, hr_tensor

    @staticmethod
    def _get_random_crop_params(img: Image.Image, output_size: int) -> Tuple[int, int, int, int]:
        w, h = img.size
        th = tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = int(torch.randint(0, h - th + 1, size=(1,)).item())
        j = int(torch.randint(0, w - tw + 1, size=(1,)).item())
        return i, j, th, tw

