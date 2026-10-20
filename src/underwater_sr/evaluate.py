"""Evaluation script for quantitative and qualitative benchmarking (PSNR & SSIM)."""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from underwater_sr.model import SRCNN
from underwater_sr.dataset import UnderwaterSRDataset
from underwater_sr.metrics import calculate_psnr, calculate_ssim
from underwater_sr.utils import get_device, load_config


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_outputs: bool = False,
    output_dir: Path = Path("outputs/evaluation"),
) -> dict:
    """Run evaluation and compute mean PSNR and SSIM."""
    model.eval()
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for idx, (lr, hr) in enumerate(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)

            psnr = calculate_psnr(sr, hr)
            ssim = calculate_ssim(sr, hr)

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            if save_outputs:
                save_path = output_dir / f"test_{idx:04d}_sr.png"
                save_image(sr.squeeze(0), save_path)

    avg_psnr = sum(psnr_scores) / max(1, len(psnr_scores))
    avg_ssim = sum(ssim_scores) / max(1, len(ssim_scores))
    return {"mean_psnr": avg_psnr, "mean_ssim": avg_ssim, "num_images": len(psnr_scores)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SRCNN model on test data")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights (.pth)")
    parser.add_argument("--scale", type=int, default=None, choices=[2, 4], help="Upscaling scale factor")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-outputs", action="store_true", help="Save super-resolved test outputs")
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    eval_cfg = cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})

    scale_factor = args.scale or model_cfg.get("scale_factor", 4)
    device = get_device(args.device or eval_cfg.get("device", "auto"))

    print(f"[UW-SR] Evaluating SRCNN | Scale: {scale_factor}x | Device: {device}")

    test_hr = eval_cfg.get("test_hr_dir", "data/test/hr")
    test_lr = eval_cfg.get("test_lr_dir", "data/test/lr")
    dataset = UnderwaterSRDataset(
        hr_dir=test_hr,
        lr_dir=test_lr,
        scale_factor=scale_factor,
        patch_size=None,
        is_train=False,
    )

    if len(dataset) == 0:
        print(f"[!] Warning: No test images found in '{test_hr}'.")
        print("    Add test images to 'data/test/hr/' to run evaluation.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = SRCNN(scale_factor=scale_factor).to(device)

    ckpt_path = args.checkpoint or f"checkpoints/best_srcnn_x{scale_factor}.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[UW-SR] Loaded checkpoint from: {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint not found at '{ckpt_path}'. Running with initial weights.")

    out_dir = Path(eval_cfg.get("output_dir", "outputs/evaluation"))
    metrics = evaluate(model, dataloader, device, save_outputs=args.save_outputs, output_dir=out_dir)

    print("\n======== Evaluation Results ========")
    print(f" Images evaluated: {metrics['num_images']}")
    print(f" Mean PSNR:        {metrics['mean_psnr']:.2f} dB")
    print(f" Mean SSIM:        {metrics['mean_ssim']:.4f}")
    print("=====================================")


if __name__ == "__main__":
    main()

