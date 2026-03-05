# Approach 1: Classical Augmentation

## What It Does

Applies traditional audio signal processing techniques to expand the training dataset:
- **Time Stretching**: Speeds up (1.1x) or slows down (0.9x) audio without changing pitch
- **Pitch Shifting**: Raises/lowers pitch by ±2 semitones without changing speed
- **Gaussian Noise**: Adds random noise for robustness to noisy environments
- **Volume Perturbation**: Randomly scales amplitude between 0.8x and 1.2x

These transformations are applied to **training audio** before Mel-spectrogram extraction, simulating natural variation in real speech conditions. Validation and test sets remain **unchanged** for fair comparison.

---

## 🚀 Quick Start

### Run Everything (Generation + Training)
```bash
cd research_pipeline/augmentation/classical
python run_classical_pipeline.py --epochs 100
```

### Run Generation Only (Create Augmented Data)
```bash
python generate_classical_augmented_data.py
```

### Run Training Only (Train Models on Existing Data)
```bash
python train_classical_models.py --epochs 100
```

### Common Options
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--skip_existing`: Skip already-trained models
- `--force`: Force regenerate/retrain
- `--stage generation|training|both`: Run specific stage (pipeline only)

---

## Data Requirements

**Input:**
- `Dataset/my Dataset/` - Original raw `.wav` files organized by emotion (Phase 1 baseline)
- One of 8 emotions: anger, disgust, fear, happy, neutral, sad, sarcastic, surprise

**Generated:**
- `research_pipeline/data/classical_augmented_features.npz` - Combined original (2240 samples) + augmented training features (~11,200 total training samples)
- Validation: 480 samples (unchanged)
- Test: 480 samples (unchanged)

## GPU Usage

- **Generation stage**: CPU only (librosa audio processing is efficient on CPU)
- **Training stage**: GPU (same as Phase 1 baseline—8 models × ~2 hours each)

## Two-Stage Pipeline

### Stage 1: Generation (Data Augmentation)

Generates augmented training features independently:

```bash
cd research_pipeline/augmentation/classical
python generate_classical_augmented_data.py
```

**What it produces:**
- `data/classical_augmented_features.npz` (compatible with Phase 1 models)
- Console output: augmentation summary and statistics
- Reproducible with random_seed=42

**Typical runtime:** 5-15 minutes (CPU)

### Stage 2: Training (Model Comparison)

Trains all 6 models on augmented features independently:

```bash
cd research_pipeline/augmentation/classical
python train_classical_models.py [--epochs 100] [--batch_size 32] [--skip_existing]
```

**What it produces:**
- `results/augmentation/classical/cnn/` - CNN + metrics
- `results/augmentation/classical/lstm/` - LSTM + metrics
- `results/augmentation/classical/cnn_lstm/` - CNN-LSTM + metrics
- `results/augmentation/classical/resnet/` - ResNet + metrics
- `results/augmentation/classical/transformer/` - Transformer + metrics
- `results/augmentation/classical/svm/` - SVM + metrics
- `results/augmentation/classical/comparison/` - All comparison visualizations

**Typical runtime:** 30-90 minutes (GPU dependent)

### Run Both Stages Together

Orchestrates generation → training:

```bash
cd research_pipeline/augmentation/classical
python run_classical_pipeline.py [--stage both|generation|training] [--epochs 100] [--force]
```

**Options:**
- `--stage`: `generation`, `training`, or `both` (default: both)
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: (default: 32)
- `--skip_existing`: Skip already-trained models
- `--force`: Regenerate/retrain even if files exist

## Output Files

```
research_pipeline/
├── data/
│   └── classical_augmented_features.npz        [Stage 1 output]
└── results/augmentation/classical/
    ├── cnn/
    │   ├── best_cnn_model.h5
    │   ├── cnn_results.json
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   └── per_class_metrics.png
    ├── lstm/, cnn_lstm/, resnet/, transformer/, svm/     [Same structure]
    └── comparison/
        ├── comparison_results.json           [Summary scores]
        ├── model_comparison.png              [Bar chart: Accuracy/F1]
        ├── training_curves_comparison.png    [Line plots]
        └── metrics_radar_chart.png           [Radar plot]
```

## Comparison with Baseline

Compare results side-by-side:
- **Baseline (no augmentation)**: `results/comparison/comparison_results.json`
- **Classical**: `results/augmentation/classical/comparison/comparison_results.json`

Key metrics:
- Test Accuracy
- Macro F1-Score
- Weighted F1-Score
- Per-class improvements (emotion-specific)

## Reproducibility

- **Random seed**: 42 (all operations)
- **Train/val/test split**: Stratified (identical to Phase 1)
- **Augmentation params**: Time stretch (0.9–1.1), pitch shift (–2 to +2), noise (0.005), volume (0.8–1.2)
- **Samples per class**: 4 augmentations per training sample
