# Synthetic Data Augmentation Guide

This guide explains how to run the **three augmentation approaches** (from Presentation Outline slides 9–12) and compare them to the **original baseline** using the same evaluation pipeline.

## Prerequisites

- **Baseline done:** Either run the baseline pipeline once so that `data/baseline_features.npz` exists, or the classical/transformer/vae pipelines will use the dataset directly (classical) or run baseline extraction first (transformer/vae).
- **Dataset:** `Dataset/my Dataset/` with 8 emotion folders and `.wav` files (same as baseline).

## Approaches at a Glance

| Approach | What it does | Data output | Results location |
|----------|--------------|-------------|------------------|
| **1. Classical** | Time stretch, pitch shift, noise, volume on **audio** → then Mel extraction | `data/classical_augmented_features.npz` | `results/augmentation/classical/` |
| **2. Transformer (MAE)** | Train MAE on spectrograms, generate new spectrograms from latent | `data/transformer_augmented_features.npz` | `results/augmentation/transformer/` |
| **3. VAE** | Train VAE on spectrograms, sample z and decode to new spectrograms | `data/vae_augmented_features.npz` | `results/augmentation/vae/` |

In all cases, **validation and test sets are unchanged** (same as baseline) so comparisons are fair.

## How to Run

Run from the **research_pipeline** directory (or from the listed subdirectory).

### 1. Classical augmentation

```bash
cd research_pipeline/augmentation/classical
python run_classical_pipeline.py
```

- Extracts features from original + augmented audio (time stretch, pitch shift, noise, volume).
- Trains all 6 models (CNN, LSTM, CNN-LSTM, ResNet, Transformer, SVM) on the augmented train set.
- Saves results under `research_pipeline/results/augmentation/classical/`.

Options: `--skip_existing`, `--force`, `--epochs`, `--batch_size`.

### 2. Transformer (MAE) augmentation

```bash
cd research_pipeline/augmentation/transformer
python run_transformer_pipeline.py
```

- Uses `data/baseline_features.npz` (creates it via baseline extraction if missing).
- Trains the MAE on training spectrograms, then generates new spectrograms.
- Builds `data/transformer_augmented_features.npz` and runs the same 6-model comparison.
- Results: `results/augmentation/transformer/`.

Options: `--mae_epochs`, `--epochs`, `--skip_existing`, `--force`.

### 3. VAE augmentation

```bash
cd research_pipeline/augmentation/vae_gan
python run_vae_pipeline.py
```

- Uses `data/baseline_features.npz` (creates it if missing).
- Trains the VAE on training spectrograms, then generates new spectrograms.
- Builds `data/vae_augmented_features.npz` and runs the same 6-model comparison.
- Results: `results/augmentation/vae/`.

Options: `--vae_epochs`, `--epochs`, `--skip_existing`, `--force`.

## Comparing with the Original Baseline

The **same evaluation protocol** is used everywhere:

- Same 6 models (CNN, LSTM, CNN-LSTM, ResNet, Transformer, SVM).
- Same metrics (accuracy, macro F1, weighted F1, per-class metrics).
- Same test set (no augmentation on test).

### Where to find numbers

- **Baseline (no augmentation):**  
  `research_pipeline/results/comparison/comparison_results.json`  
  and per-model: `results/cnn/cnn_results.json`, etc.

- **Classical:**  
  `research_pipeline/results/augmentation/classical/comparison/comparison_results.json`  
  and `results/augmentation/classical/<model>/<model>_results.json`.

- **Transformer:**  
  `research_pipeline/results/augmentation/transformer/comparison/comparison_results.json`.

- **VAE:**  
  `research_pipeline/results/augmentation/vae/comparison/comparison_results.json`.

In each `comparison_results.json`, use the `summary` (or equivalent) block for accuracy and macro F1 to compare baseline vs classical vs transformer vs VAE.

### Suggested comparison table

| Setting | Best model (e.g. by macro F1) | Test accuracy | Macro F1 |
|---------|------------------------------|---------------|----------|
| Baseline | … | … | … |
| Classical | … | … | … |
| Transformer (MAE) | … | … | … |
| VAE | … | … | … |

Fill this from the JSONs above for your report or presentation.

## Resuming and Overwriting

- **Skip already-trained models:** use `--skip_existing` (per approach).
- **Retrain everything for an approach:** use `--force` (and optionally delete that approach’s `data/*_augmented_features.npz` to regenerate augmented data).

## Directory Layout (augmentation)

```
research_pipeline/
├── data/
│   ├── baseline_features.npz
│   ├── classical_augmented_features.npz
│   ├── transformer_augmented_features.npz
│   └── vae_augmented_features.npz
├── results/
│   ├── comparison/                    # Baseline comparison
│   └── augmentation/
│       ├── classical/
│       │   ├── cnn/, lstm/, ... comparison/
│       ├── transformer/
│       │   ├── mae_weights/, cnn/, ... comparison/
│       └── vae/
│           ├── vae_weights/, cnn/, ... comparison/
└── augmentation/
    ├── README.md
    ├── classical/
    ├── transformer/
    └── vae_gan/
```

For more detail per approach, see `augmentation/README.md` and the README in each of `classical/`, `transformer/`, and `vae_gan/`.
