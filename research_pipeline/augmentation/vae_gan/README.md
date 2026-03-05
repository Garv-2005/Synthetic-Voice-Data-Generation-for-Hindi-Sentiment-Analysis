# Approach 3: VAE (Variational Autoencoder) Spectrogram Generation

## What It Does

Learns the latent distribution of speech spectrograms using a Variational Autoencoder (VAE), then generates novel spectrograms by sampling from the learned latent space.

**Process:**
1. **Encode**: Compress each spectrogram to a latent vector with mean μ and variance σ
2. **Reparameterize**: Learn a Gaussian distribution N(μ, σ) in latent space (via KL divergence regularization)
3. **Decode**: Reconstruct spectrograms from latent vectors
4. **Generate**: Sample z ~ N(0,1) and decode to generate new spectrograms

This probabilistic approach ensures generated samples lie in a well-defined latent distribution, potentially more stable and diverse than MAE. The KL divergence regularization prevents "posterior collapse" and maintains meaningful latent structure.

---

## 🚀 Quick Start

### Run Everything (Generation + Training)
```bash
cd research_pipeline/augmentation/vae_gan
python run_vae_pipeline.py --epochs 100 --vae_epochs 50
```

### Run Generation Only (Create Augmented Data)
```bash
python generate_vae_augmented_data.py --vae_epochs 50
```

### Run Training Only (Train Models on Existing Data)
```bash
python train_vae_models.py --epochs 100
```

### Common Options
- `--vae_epochs`: VAE training epochs (default: 50, increase for better quality)
- `--epochs`: Model training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--num_generated_per_class`: Synthetic samples per class (default: 50)
- `--skip_existing`: Skip already-trained models
- `--force`: Force regenerate/retrain

---

## Data Requirements

**Input:**
- `research_pipeline/data/baseline_features.npz` (from Phase 1 baseline extraction)
  - If missing, generation stage automatically extracts from `Dataset/my Dataset/`
- Contains: X_train (2240 samples), X_val (480), X_test (480), emotion labels

**Generated:**
- `research_pipeline/data/vae_augmented_features.npz` - Original + VAE-generated spectrograms
- `research_pipeline/results/augmentation/vae/vae_weights/vae_weights.keras` - Trained VAE weights

## GPU Usage

- **Generation stage**: GPU required
  - VAE training: ~10-60 minutes (depending on vae_epochs, typically 50)
  - Spectrogram generation: ~2-5 minutes
- **Training stage**: GPU (same as Phase 1 baseline)

**Recommended**: Run on GPU for reasonable timelines

## Two-Stage Pipeline

### Stage 1: Generation (VAE Training + Sampling)

Generates synthetic spectrograms independently:

```bash
cd research_pipeline/augmentation/vae_gan
python generate_vae_augmented_data.py [--vae_epochs 50] [--batch_size 32] [--num_generated_per_class 50]
```

**What it produces:**
1. Ensures baseline features exist (extracts if needed)
2. Trains VAE on baseline training spectrograms
   - Saves: `results/augmentation/vae/vae_weights/vae_weights.keras`
   - Checkpoint: best weights during training
   - Loss: Reconstruction + KL divergence
3. Samples from latent distribution (z ~ N(0,1))
4. Decodes to generate synthetic spectrograms
   - 50 samples per emotion class (400 total) by default
5. Combines with original training data
   - Saves: `data/vae_augmented_features.npz`

**Typical runtime:** 20-60 minutes (GPU)

**Key parameters:**
- `--vae_epochs`: VAE training epochs (default: 50, increase for better reconstruction)
- `--num_generated_per_class`: Synthetic samples per emotion (default: 50)
- `--batch_size`: VAE batch size (default: 32)

### Stage 2: Training (Model Comparison)

Trains all 6 models on VAE-augmented features independently:

```bash
cd research_pipeline/augmentation/vae_gan
python train_vae_models.py [--epochs 100] [--batch_size 32] [--skip_existing]
```

**What it produces:**
- Models (CNN, LSTM, CNN-LSTM, ResNet, Transformer, SVM) + results
- `results/augmentation/vae/comparison/` - All comparison visualizations

**Typical runtime:** 30-90 minutes (GPU)

### Run Both Stages Together

Orchestrates generation → training:

```bash
cd research_pipeline/augmentation/vae_gan
python run_vae_pipeline.py [--stage both|generation|training] [--vae_epochs 50] [--epochs 100] [--force]
```

**Options:**
- `--stage`: `generation`, `training`, or `both` (default: both)
- `--vae_epochs`: VAE training epochs (default: 50)
- `--epochs`: Model training epochs (default: 100)
- `--batch_size`: (default: 32)
- `--num_generated_per_class`: Synthetic samples per class (default: 50)
- `--skip_existing`: Skip already-trained models
- `--force`: Regenerate/retrain even if files exist

## Output Files

```
research_pipeline/
├── data/
│   └── vae_augmented_features.npz               [Stage 1 output]
└── results/augmentation/vae/
    ├── vae_weights/
    │   └── vae_weights.keras                     [Trained VAE encoder-decoder]
    ├── cnn/
    │   ├── best_cnn_model.h5
    │   ├── cnn_results.json
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   └── per_class_metrics.png
    ├── lstm/, cnn_lstm/, resnet/, transformer/, svm/     [Same structure]
    └── comparison/
        ├── comparison_results.json           [Summary scores]
        ├── model_comparison.png              [Bar chart]
        ├── training_curves_comparison.png    [Line plots]
        └── metrics_radar_chart.png           [Radar plot]
```

## Comparison with Baseline

- **Baseline (no augmentation)**: `results/comparison/comparison_results.json`
- **VAE**: `results/augmentation/vae/comparison/comparison_results.json`

Key metrics:
- Test Accuracy
- Macro F1-Score
- Weighted F1-Score
- Per-class improvements

## How VAE Improves Over Classical and MAE

- **Probabilistic**: Well-defined latent distribution (z ~ N(0,1)) ensures diversity
- **Principled generation**: Samples from theoretical distribution, not heuristics
- **Regularization**: KL divergence prevents overfitting of latent space
- **Stable**: Less prone to mode collapse than GAN approaches
- Trade-off: More sensitive to hyperparameters (epochs, latent dimension, β coefficient)

## VAE Architecture Details

- **Input shape**: (128, 174, 1) - Mel-spectrogram
- **Encoder**: Convolutional layers → latent (μ, σ)
- **Decoder**: Latent → deconvolutional layers → (128, 174, 1)
- **Loss**: Reconstruction loss (MSE) + KL divergence
- **Latent dimension**: Configured in `vae_spectrogram.py`

## Reproducibility

- **Random seed**: 42 (all operations)
- **Train/val/test split**: Identical to Phase 1
- **VAE loss**: Reconstruction (MSE) + KL(N(μ,σ)||N(0,1))
- **Number of VAE samples per class**: 50 (configurable)
- **Sampling strategy**: z ~ N(0,1), decode deterministically

## Future Enhancement: GAN

Placeholder structure exists for GAN-based generation (optional future work).
