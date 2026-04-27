# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Advanced ML course project comparing CNN architectures for image classification. The core research question is whether **dilated convolutions with learnable dilation weights** (DilatedCNN) outperform a standard CNN baseline on MNIST.

## Running the Project

```bash
python project.py
```

No build step required. Runs 5 training iterations of 12 epochs each and reports accuracy stats (average, best, worst).

## Dependencies

No `requirements.txt` exists. Required packages:
- `torch`, `torchvision`
- `torchmetrics`
- `numpy`, `pandas`, `matplotlib`
- `tqdm`

## Architecture

### Models (`models/`)

**`baseline_cnn.py`** — Standard CNN with two conv layers (8→16 channels, 3×3 kernels), MaxPool after each, and a single FC layer mapping `16×7×7 → 10`. Unused pooling variants (AvgPool2d, LPPool2d) are defined but not wired into the forward pass.

**`dilated_cnn.py`** — The novel architecture. Each conv layer runs three parallel dilated convolutions (dilation rates 1, 2, 3) over a shared base kernel, then combines them via a **learnable softmax-weighted sum**. The `alpha1`/`alpha2` parameters are the trainable mixing weights. Uses `F.conv2d` directly with explicit padding to handle varying dilation rates.

**`test_cnn.py`** — A cleaner, more modular reimplementation of the dilated approach using a `MultiDilatedConv` class with separate `nn.Conv2d` layers per dilation rate. Not imported by `project.py` but architecturally equivalent and easier to extend.

### Training (`project.py`)

- Both models run every execution; results for each are printed at the end
- Batch size: 10, Adam optimizer (lr=0.0005), CrossEntropyLoss
- 5 independent runs of 5 epochs each; reports per-run and aggregate metrics
- **Switch datasets** by changing `DATASET = "MNIST"` near the top of `project.py` — options: `"MNIST"`, `"CIFAR10"`, `"STL10"`
- Models auto-adjust their FC layer size via the `img_size` parameter passed from `DATASET_CONFIGS`

### Datasets (`dataset/`)

- **MNIST** — downloaded and ready at `dataset/MNIST/raw/`
- **CIFAR-10** — downloaded and extracted at `dataset/cifar-10-batches-py/` but not used in `project.py`

## Key Design Decisions

The DilatedCNN's learnable `alpha` weights (one per channel per dilation rate) let the model discover which receptive field size is most useful per channel, rather than committing to a fixed dilation. Softmax normalization keeps the weighted sum stable during training.
