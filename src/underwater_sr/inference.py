"""Inference script for running super-resolution on single images or folders."""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch

from underwater_sr.model import SRCNN
from underwater_sr.utils import get_device, pil_to_tensor, tensor_to_pil, load_config


def super_resolve_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> Image.Image:
    """Super-resolve a single PIL RGB image."""
    model.eval()
    lr_tensor = pil_to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    return tensor_to_pil(sr_tensor.squeeze(0))


def run_inference(
    input_path: str,
    output_dir: str,
    checkpoint_path: str = None,
    scale_factor: int = 4,
    device_pref: str = "auto",
) -> list[str]:
    """Run super-resolution inference on an image file or directory of images."""
    device = get_device(device_pref)
    model = SRCNN(scale_factor=scale_factor).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[UW-SR] Loaded checkpoint from: {checkpoint_path}")
    else:
        print("[!] Running inference without trained weights (untrained model).")

    input_p = Path(input_path)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    if input_p.is_file():
        image_files = [input_p]
    elif input_p.is_dir():
        image_files = [
            p for p in input_p.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        ]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    saved_paths = []
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        sr_img = super_resolve_image(model, img, device)

        out_name = f"{img_file.stem}_sr_x{scale_factor}.png"
        out_file = output_p / out_name
        sr_img.save(out_file)
        saved_paths.append(str(out_file))
        print(f"[UW-SR] Saved: {out_file} ({img.size} -> {sr_img.size})")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Super-resolution inference on underwater images")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or image directory")
    parser.add_argument("--output", type=str, default="outputs/inference", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4], help="Upscaling factor (2 or 4)")
    parser.add_argument("--device", type=str, default="auto", help="Device preference (cpu, cuda, auto)")
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        scale_factor=args.scale,
        device_pref=args.device,
    )


if __name__ == "__main__":
    main()

