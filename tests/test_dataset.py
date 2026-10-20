"""Unit tests for dataset loading and preprocessing."""

import tempfile
from pathlib import Path
from PIL import Image
import torch
from underwater_sr.dataset import UnderwaterSRDataset


def test_dataset_synthetic_downscaling():
    """Verify dataset generates downscaled pairs when only HR images are given."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hr_dir = Path(tmpdir) / "hr"
        hr_dir.mkdir()

        # Create two synthetic dummy HR images
        img1 = Image.new("RGB", (128, 128), color=(20, 80, 140))
        img2 = Image.new("RGB", (128, 128), color=(40, 100, 160))
        img1.save(hr_dir / "test1.png")
        img2.save(hr_dir / "test2.png")

        dataset = UnderwaterSRDataset(
            hr_dir=str(hr_dir),
            scale_factor=4,
            patch_size=64,
            is_train=True,
        )

        assert len(dataset) == 2

        lr_t, hr_t = dataset[0]
        assert isinstance(lr_t, torch.Tensor)
        assert isinstance(hr_t, torch.Tensor)
        assert hr_t.shape == (3, 64, 64)
        assert lr_t.shape == (3, 16, 16)

