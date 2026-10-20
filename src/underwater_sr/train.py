"""Training script for Underwater Image Super-Resolution."""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from underwater_sr.model import SRCNN
from underwater_sr.dataset import UnderwaterSRDataset
from underwater_sr.metrics import calculate_psnr
from underwater_sr.utils import get_device, load_config


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0

    for lr, hr in dataloader:
        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * lr.size(0)

    return running_loss / max(1, len(dataloader.dataset))


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return (val_loss, mean_psnr)."""
    model.eval()
    running_loss = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for lr, hr in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)
            loss = criterion(sr, hr)

            running_loss += loss.item() * lr.size(0)
            total_psnr += calculate_psnr(sr, hr) * lr.size(0)
            count += lr.size(0)

    val_loss = running_loss / max(1, count)
    val_psnr = total_psnr / max(1, count)
    return val_loss, val_psnr


def main():
    parser = argparse.ArgumentParser(description="Train SRCNN for Underwater Image Super-Resolution")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--scale", type=int, default=None, choices=[2, 4], help="Upscaling factor")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    scale_factor = args.scale or model_cfg.get("scale_factor", 4)
    epochs = args.epochs or train_cfg.get("epochs", 50)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    lr = args.lr or train_cfg.get("learning_rate", 1e-4)
    device_pref = args.device or train_cfg.get("device", "auto")
    device = get_device(device_pref)

    print(f"[UW-SR] Training SRCNN | Scale: {scale_factor}x | Device: {device} | Epochs: {epochs}")

    # Datasets
    train_hr = data_cfg.get("train_hr_dir", "data/train/hr")
    train_lr = data_cfg.get("train_lr_dir", "data/train/lr")
    val_hr = data_cfg.get("val_hr_dir", "data/val/hr")
    val_lr = data_cfg.get("val_lr_dir", "data/val/lr")

    train_ds = UnderwaterSRDataset(
        hr_dir=train_hr,
        lr_dir=train_lr,
        scale_factor=scale_factor,
        patch_size=data_cfg.get("patch_size", 128),
        is_train=True,
    )

    val_ds = UnderwaterSRDataset(
        hr_dir=val_hr,
        lr_dir=val_lr,
        scale_factor=scale_factor,
        patch_size=None,
        is_train=False,
    )

    if len(train_ds) == 0:
        print(f"[!] Warning: No images found in '{train_hr}'.")
        print("    Place HR training images in 'data/train/hr/' or check configs/config.yaml.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False) if len(val_ds) > 0 else None

    # Model, Loss, Optimizer
    model = SRCNN(scale_factor=scale_factor).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg.get("lr_step_size", 20),
        gamma=train_cfg.get("lr_gamma", 0.5),
    )

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_psnr = -1.0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_info = ""
        if val_loader:
            v_loss, v_psnr = validate(model, val_loader, criterion, device)
            val_info = f" | Val Loss: {v_loss:.5f} | Val PSNR: {v_psnr:.2f} dB"
            if v_psnr > best_psnr:
                best_psnr = v_psnr
                best_path = ckpt_dir / f"best_srcnn_x{scale_factor}.pth"
                torch.save(model.state_dict(), best_path)
                val_info += " [Best Saved]"

        print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {loss:.5f}{val_info}")

        if epoch % train_cfg.get("save_every", 10) == 0:
            torch.save(model.state_dict(), ckpt_dir / f"srcnn_x{scale_factor}_epoch_{epoch:03d}.pth")

    print("[UW-SR] Training process finished.")


if __name__ == "__main__":
    main()

