# Approach 2: Transformer-Based (MAE) Spectrogram Generation

## What It Does

Learns the structure of speech spectrograms using a Masked Autoencoder (MAE), then generates novel spectrograms by sampling from the learned representation.

**Process:**
1. **Tokenize**: Split each spectrogram into 16×18 patches, flatten into a sequence
2. **Mask**: Randomly mask ~75% of patches during training (self-supervised learning)
3. **Train**: Transformer encoder-decoder learns to reconstruct masked patches
4. **Generate**: Sample / iteratively generate new spectrograms from the learned latent space

This approach learns meaningful feature representations directly from the Mel-spectrograms, potentially capturing emotion-specific patterns in the spectrogram structure.

---

## 🚀 Quick Start

### Run Everything (Generation + Training)
```bash
cd research_pipeline/augmentation/transformer
python run_transformer_pipeline.py --epochs 100 --mae_epochs 30
```

### Run Generation Only (Create Augmented Data)
```bash
python generate_transformer_augmented_data.py --mae_epochs 30
```

### Run Training Only (Train Models on Existing Data)
```bash
python train_transformer_models.py --epochs 100
```

### Common Options
- `--mae_epochs`: MAE training epochs (default: 30, increase for better quality)
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
- Contains: X_train (2240 samples), X_val (480), X_test (480), emotions labels

**Generated:**
- `research_pipeline/data/transformer_augmented_features.npz` - Original + MAE-generated spectrograms
- `research_pipeline/results/augmentation/transformer/mae_weights/mae_weights.keras` - Trained MAE weights

## GPU Usage

- **Generation stage**: GPU required
  - MAE training: ~5-30 minutes (depending on mae_epochs, typically 30)
  - Spectrogram generation: ~2-5 minutes
- **Training stage**: GPU (same as Phase 1 baseline)

**Recommended**: Run on GPU for reasonable timelines

## Two-Stage Pipeline

### Stage 1: Generation (MAE Training + Synthetic Spectrogram Generation)

Generates synthetic spectrograms independently:

```bash
cd research_pipeline/augmentation/transformer
python generate_transformer_augmented_data.py [--mae_epochs 30] [--batch_size 32] [--num_generated_per_class 50]
```

**What it produces:**
1. Ensures baseline features exist (extracts if needed)
2. Trains MAE on baseline training spectrograms
   - Saves: `results/augmentation/transformer/mae_weights/mae_weights.keras`
   - Checkpoint: best weights during training
3. Generates synthetic spectrograms using trained MAE
   - 50 samples per emotion class (400 total) by default
4. Combines with original training data
   - Saves: `data/transformer_augmented_features.npz`

**Typical runtime:** 15-45 minutes (GPU)

**Key parameters:**
- `--mae_epochs`: MAE training epochs (default: 30, increase for better quality)
- `--num_generated_per_class`: Synthetic samples per emotion (default: 50)
- `--batch_size`: MAE batch size (default: 32)

### Stage 2: Training (Model Comparison)

Trains all 6 models on MAE-augmented features independently:

```bash
cd research_pipeline/augmentation/transformer
python train_transformer_models.py [--epochs 100] [--batch_size 32] [--skip_existing]
```

**What it produces:**
- Models (CNN, LSTM, CNN-LSTM, ResNet, Transformer, SVM) + results
- `results/augmentation/transformer/comparison/` - All comparison visualizations

**Typical runtime:** 30-90 minutes (GPU)

### Run Both Stages Together

Orchestrates generation → training:

```bash
cd research_pipeline/augmentation/transformer
python run_transformer_pipeline.py [--stage both|generation|training] [--mae_epochs 30] [--epochs 100] [--force]
```

**Options:**
- `--stage`: `generation`, `training`, or `both` (default: both)
- `--mae_epochs`: MAE training epochs (default: 30)
- `--epochs`: Model training epochs (default: 100)
- `--batch_size`: (default: 32)
- `--num_generated_per_class`: Synthetic samples per class (default: 50)
- `--skip_existing`: Skip already-trained models
- `--force`: Regenerate/retrain even if files exist

## Output Files

```
research_pipeline/
├── data/
│   └── transformer_augmented_features.npz        [Stage 1 output]
└── results/augmentation/transformer/
    ├── mae_weights/
    │   └── mae_weights.keras                      [Trained MAE encoder-decoder]
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
- **Transformer (MAE)**: `results/augmentation/transformer/comparison/comparison_results.json`

Key metrics:
- Test Accuracy
- Macro F1-Score
- Weighted F1-Score
- Per-class improvements

## How MAE Improves Over Classical

- **Self-supervised learning**: Learns representations from spectrogram structure, not just signal transformations
- **Realistic spectrograms**: Generated samples preserve Mel-spectrogram characteristics
- **Adaptive augmentation**: Learns what variations matter for the specific dataset
- But may be slower and more complex than classical approaches

## Reproducibility

- **Random seed**: 42 (all operations)
- **MAE architecture**: 4-layer encoder, 4-layer decoder, patch size 16×18
- **Masking ratio**: ~75% during training
- **Train/val/test split**: Identical to Phase 1
- **Number of MAE samples per class**: 50 (configurable)
