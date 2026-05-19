# 🌿 Plant Leaf Super Resolution

A GAN-based image super resolution pipeline that upscales low-resolution (LR) plant leaf images to high-resolution (HR) at 4× scale.

> **Note:** The dataset used is from a private Kaggle competition and is not included in this repository. Place your data under a `data/` folder following the structure below.

```
data/
├── train_High_Resolution/
├── train_Low_Resolution/
└── test_Low_Resolution/
```

---

## Architecture

**Generator (SRGAN-style)**
- 23 residual blocks with scaled residual connections (`0.2` scaling)
- 2× PixelShuffle upsampling stages (4× total)
- `Tanh` output activation

**Discriminator (PatchGAN-style)**
- Concatenates HR/upscaled-LR image pairs as input (6 channels)
- 4-layer strided convolution network for patch-level real/fake classification

**Perceptual Loss**
- VGG-19 features up to layer 18 (pre-trained, frozen)
- L1 loss in feature space

---

## Loss Function

Warmup phase (epochs 0–20) — no adversarial loss:
```
L_G = 10 × L1_pixel + 0.009 × L_perceptual
```

Main phase (epochs 21+) — adversarial loss enabled conditionally:
```
L_G = 10 × L1_pixel + 0.1 × L_perceptual + adv_weight × L_adversarial
# adv_weight = 0.005 if D_loss > 0.35 else 0.0
```

---

## Training

| Setting | Value |
|---|---|
| Epochs | 150 |
| Batch size | 16 |
| G learning rate | 1e-4 |
| D learning rate | 1e-6 |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Mixed precision | AMP (`autocast` + `GradScaler`) |
| Hardware | NVIDIA Tesla T4 (GPU) |

Best model checkpoint saved based on lowest validation MAE.

**Data Augmentation:** random horizontal/vertical flips and 90° rotations applied during training.

---

## Inference

Test-Time Augmentation (TTA) averages predictions across 8 augmentations — 4 rotations × 2 flips — for improved output quality.

Output is a `submission.csv` with flattened pixel values per image.

---

## Project Structure

```
├── CropsDataset        # Dataset class (LR/HR pairs or LR-only)
├── ResidualBlock       # Residual block with scaled skip connection
├── Generator           # SRGAN-style upscaling network
├── Discriminator       # PatchGAN discriminator
├── PerceptualLoss      # VGG-19 feature loss
└── submission.csv      # Final predictions
```

---

## Requirements

```
torch, torchvision, Pillow, numpy, pandas
```**
