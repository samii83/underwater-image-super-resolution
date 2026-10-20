# Underwater Datasets Directory

This folder holds datasets for underwater image super-resolution.

## Recommended Datasets

1. **UIEB (Underwater Image Enhancement Benchmark)**
   - 890 paired real-world underwater images with reference images.
   - [Paper & Dataset](https://li-chongyi.github.io/proj_benchmark.html)
2. **EUVP (Enhancing Underwater Visual Perception)**
   - Over 12,000 paired and unpaired underwater images captured under varying oceanic conditions.
3. **UFO-120 (Underwater Four-in-One)**
   - Benchmark specifically featuring underwater super-resolution pairs with 2x, 3x, and 4x scale factors.

## Expected Directory Layout

```text
data/
├── train/
│   ├── hr/     # High-resolution training images (*.png, *.jpg)
│   └── lr/     # (Optional) Pre-generated low-resolution images
├── val/
│   ├── hr/     # High-resolution validation images
│   └── lr/     # (Optional) Pre-generated low-resolution images
└── test/
    ├── hr/     # High-resolution test benchmark images
    └── lr/     # (Optional) Pre-generated low-resolution images
```

> [!NOTE]
> If `lr/` folders are omitted, `UnderwaterSRDataset` will automatically synthesize low-resolution inputs via bicubic downsampling.

