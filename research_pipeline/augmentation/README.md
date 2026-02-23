# Synthetic Data Augmentation for Hindi SER

This directory implements **three augmentation approaches** (Slides 9–12 of the presentation outline) so you can compare them against the **same baseline evaluation pipeline** (same models, same metrics, same test set).

## Directory layout

```
augmentation/
├── README.md                 # This file
├── classical/                # Approach 1: Classical audio augmentations
├── transformer/              # Approach 2: Transformer-based (MAE) spectrogram generation
└── vae_gan/                  # Approach 3: VAE (and GAN) spectrogram generation
```

Each approach has:
- **Feature/generation scripts** that produce an `.npz` file in the **same format** as `../data/baseline_features.npz`.
- **Run script** that builds augmented data (if needed) and runs the **same 6 models** (CNN, LSTM, CNN-LSTM, ResNet, Transformer, SVM) via the existing comparison pipeline.
- **Results** written under `../results/augmentation/<approach>/` so outputs are segregated and easy to compare.

## Fair comparison with baseline

- **Test and validation sets are never augmented.** Only the training set is expanded (original + synthetic/augmented).
- **Same split and emotion_map** as baseline (same random seed and split logic).
- **Same evaluation:** `comparison/train_model.py` and `comparison/compare_models.py` are used as-is; only the input `.npz` and results directory change.

## Approaches

### 1. Classical augmentation (`classical/`)

- **What:** Time stretching, pitch shifting, background noise, volume perturbation applied to **audio** before Mel-spectrogram extraction.
- **Output:** `../data/classical_augmented_features.npz`
- **Results:** `../results/augmentation/classical/` (per-model folders + `comparison/`)

### 2. Transformer-based generation (`transformer/`)

- **What:** Masked Autoencoder (MAE) on spectrograms; generate new spectrograms from the learned representation.
- **Output:** `../data/transformer_augmented_features.npz`
- **Results:** `../results/augmentation/transformer/`

### 3. Advanced generative – VAE/GAN (`vae_gan/`)

- **What:** VAE (and optionally GAN) to generate new spectrograms from latent space.
- **Output:** `../data/vae_augmented_features.npz` (and optionally GAN)
- **Results:** `../results/augmentation/vae/` (and `gan/` if added)

## Quick commands

```bash
# From research_pipeline root
cd augmentation/classical
python run_classical_pipeline.py

cd ../transformer
python run_transformer_pipeline.py

cd ../vae_gan
python run_vae_pipeline.py
```

Each pipeline can take `--skip_existing` to skip already-trained models and `--epochs`, `--batch_size` as in the baseline comparison.

## Comparing results

- **Baseline (no augmentation):** `results/comparison/comparison_results.json` and `results/<model>/`
- **Classical:** `results/augmentation/classical/comparison/comparison_results.json` and `results/augmentation/classical/<model>/`
- **Transformer:** `results/augmentation/transformer/comparison/comparison_results.json`
- **VAE:** `results/augmentation/vae/comparison/comparison_results.json`

Use the same metrics (e.g. test accuracy, macro F1) across these JSONs to compare baseline vs each augmentation approach.
