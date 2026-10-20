# Model Checkpoints

This directory stores saved PyTorch weights (`*.pth` or `*.pt`).

## Naming Convention
- `best_srcnn_x2.pth`: Weights with the highest validation PSNR for 2× scale.
- `best_srcnn_x4.pth`: Weights with the highest validation PSNR for 4× scale.
- `srcnn_x{scale}_epoch_{N}.pth`: Periodic training snapshots.

## Usage
To evaluate or run inference using a checkpoint:
```bash
python -m underwater_sr.inference --checkpoint checkpoints/best_srcnn_x4.pth --input sample_images/sample.png --scale 4
```

> [!NOTE]
> Checkpoint files (`*.pth`, `*.pt`, `*.ckpt`) are excluded from Git tracking via `.gitignore`.

