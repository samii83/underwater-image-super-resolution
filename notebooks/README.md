# Notebooks

This directory contains Jupyter notebooks covering the end-to-end underwater image super-resolution workflow:

| Notebook | Purpose |
|---|---|
| [`01_dataset_exploration.ipynb`](01_dataset_exploration.ipynb) | Dataset distribution analysis, RGB channel attenuation, histograms, and color casts. |
| [`02_preprocessing_and_degradation.ipynb`](02_preprocessing_and_degradation.ipynb) | Simulating underwater optical degradation, forward scattering blur, downsampling, and paired dataset checks. |
| [`03_training_srcnn.ipynb`](03_training_srcnn.ipynb) | Step-by-step training pipeline, optimizer updates, loss monitoring, and checkpoint saving. |
| [`04_evaluation_and_benchmarking.ipynb`](04_evaluation_and_benchmarking.ipynb) | Benchmarking bicubic baseline vs SRCNN, computing PSNR and SSIM, and qualitative visual comparisons. |

## Running the Notebooks
```bash
jupyter notebook notebooks/
```

> [!NOTE]
> All `.ipynb` notebook files are tracked in Git, while `.ipynb_checkpoints/` directories are ignored.
