# Phase 2: Synthetic Data Augmentation for Hindi SER

## Overview

This directory implements **three augmentation approaches** to improve model performance through data generation. Each approach can be run independently with separated **generation** and **training** stages.

## The Three Approaches

| Approach | Technique | Data Location | GPU | Complexity |
|----------|-----------|---------------|----|------------|
| **Classical** | Audio transformations (time stretch, pitch shift, noise, volume) | `classical/` | No (CPU) | ⭐ Low |
| **Transformer (MAE)** | Masked Autoencoder on spectrograms | `transformer/` | Yes | ⭐⭐⭐ Medium |
| **VAE** | Variational Autoencoder + sampling | `vae_gan/` | Yes | ⭐⭐⭐⭐ High |

### Quick Comparison

**Classical Augmentation** ✓
- Fastest, CPU-only generation
- Based on proven audio DSP techniques
- Simple and interpretable
- **Best for:** Quick baseline, parameter tuning
- Typical time: 20-30 min total

**Transformer (MAE)** ✓
- Self-supervised learning on spectrograms
- Learns data-specific representations
- Generates realistic spectrograms
- **Best for:** Learning semantic structure
- Typical time: 1-2 hours total (with GPU)

**VAE** ✓
- Probabilistic latent distribution
- Most diverse generation
- Well-regularized sampling
- **Best for:** Diverse, stable augmentation
- Typical time: 2-3 hours total (with GPU)

## Two-Stage Architecture

Each approach is split into two independent but coordinated stages:

### Stage 1: Generation
Produces augmented `.npz` files with original + synthetic training data
- Classical: `python classical/generate_classical_augmented_data.py`
- Transformer: `python transformer/generate_transformer_augmented_data.py`
- VAE: `python vae_gan/generate_vae_augmented_data.py`

### Stage 2: Training
Trains all 6 models (CNN, LSTM, CNN-LSTM, ResNet, Transformer, SVM) using comparison framework
- Classical: `python classical/train_classical_models.py`
- Transformer: `python transformer/train_transformer_models.py`
- VAE: `python vae_gan/train_vae_models.py`

### Full Pipeline (Optional)
Orchestrates both stages sequentially:
- Classical: `python classical/run_classical_pipeline.py`
- Transformer: `python transformer/run_transformer_pipeline.py`
- VAE: `python vae_gan/run_vae_pipeline.py`

## Quick Start

### Option 1: Run All Three Approaches

```bash
# Classical (CPU)
cd classical
python run_classical_pipeline.py

# Transformer (GPU)
cd ../transformer
python run_transformer_pipeline.py --mae_epochs 30

# VAE (GPU)
cd ../vae_gan
python run_vae_pipeline.py --vae_epochs 50
```

### Option 2: Run Only Generation (Create Augmented Data)

```bash
python classical/generate_classical_augmented_data.py
python transformer/generate_transformer_augmented_data.py --mae_epochs 30
python vae_gan/generate_vae_augmented_data.py --vae_epochs 50
```

### Option 3: Run Only Training (Use Existing Data)

```bash
python classical/train_classical_models.py --epochs 100
python transformer/train_transformer_models.py --epochs 100
python vae_gan/train_vae_models.py --epochs 100
```

### Option 4: Run One Approach at a Time

```bash
cd classical
python run_classical_pipeline.py --stage both --epochs 100 --skip_existing
```

## Data Flow

```
Dataset: my Dataset/ (raw .wav files)
    ├→ Classical Augmentation
    │   ├─ Stage 1: Audio transformations → classical_augmented_features.npz
    │   └─ Stage 2: Train models → results/augmentation/classical/comparison/
    │
    ├→ Transformer (MAE)
    │   ├─ Requires: baseline_features.npz (auto-extracts if missing)
    │   ├─ Stage 1: Train MAE → generate → transformer_augmented_features.npz
    │   └─ Stage 2: Train models → results/augmentation/transformer/comparison/
    │
    └→ VAE
        ├─ Requires: baseline_features.npz (auto-extracts if missing)
        ├─ Stage 1: Train VAE → sample → vae_augmented_features.npz
        └─ Stage 2: Train models → results/augmentation/vae/comparison/
```

## Output Structure

```
research_pipeline/
├── data/
│   ├── baseline_features.npz                    [Phase 1]
│   ├── classical_augmented_features.npz         [Approach 1, Stage 1]
│   ├── transformer_augmented_features.npz       [Approach 2, Stage 1]
│   └── vae_augmented_features.npz               [Approach 3, Stage 1]
│
└── results/augmentation/
    ├── classical/
    │   ├── cnn/, lstm/, cnn_lstm/, resnet/, transformer/, svm/
    │   └── comparison/
    │       ├── comparison_results.json
    │       ├── model_comparison.png
    │       ├── training_curves_comparison.png
    │       └── metrics_radar_chart.png
    ├── transformer/
    │   ├── mae_weights/mae_weights.keras
    │   ├── cnn/, lstm/, cnn_lstm/, resnet/, transformer/, svm/
    │   └── comparison/
    │       └── [same comparison files]
    └── vae/
        ├── vae_weights/vae_weights.keras
        ├── cnn/, lstm/, cnn_lstm/, resnet/, transformer/, svm/
        └── comparison/
            └── [same comparison files]
```

## Comparing with Baseline

All three approaches use the same evaluation framework as Phase 1:

- **Baseline (Phase 1)**: `results/comparison/comparison_results.json`
- **Classical**: `results/augmentation/classical/comparison/comparison_results.json`
- **Transformer**: `results/augmentation/transformer/comparison/comparison_results.json`
- **VAE**: `results/augmentation/vae/comparison/comparison_results.json`

Each contains:
```json
{
  "summary": {
    "CNN": {"accuracy": ..., "macro_f1": ..., "weighted_f1": ...},
    "LSTM": {...},
    "CNN-LSTM": {...},
    "ResNet": {...},
    "Transformer": {...},
    "SVM": {...}
  }
}
```

## Common Commands & Usage

### Run Everything

```bash
# Navigate to augmentation folder
cd augmentation

# Run all three approaches
python classical/run_classical_pipeline.py
python transformer/run_transformer_pipeline.py --mae_epochs 30
python vae_gan/run_vae_pipeline.py --vae_epochs 50
```

### Run Specific Approach - Both Stages

```bash
cd augmentation/classical
python run_classical_pipeline.py --stage both --epochs 100
```

### Run Generation Only (Skip Training)

```bash
cd augmentation/transformer
python generate_transformer_augmented_data.py --mae_epochs 30
```

### Run Training Only (Skip Generation)

```bash
cd augmentation/vae_gan
python train_vae_models.py --epochs 100
```

### Skip Already-Trained Models

```bash
python train_classical_models.py --skip_existing
```

### Force Regenerate/Retrain

```bash
python run_vae_pipeline.py --stage generation --force  # Regenerate VAE data
python run_vae_pipeline.py --stage training --force    # Retrain all 6 models
```

### Customize Parameters

```bash
# More MAE epochs for better quality
python transformer/generate_transformer_augmented_data.py --mae_epochs 50

# More synthetic samples per class
python vae_gan/generate_vae_augmented_data.py --num_generated_per_class 100

# Longer model training
python classical/train_classical_models.py --epochs 150 --batch_size 16
```

## GPU & Hardware

- **Classical Generation**: CPU only (fast, ~5-15 min)
- **Transformer Generation**: GPU recommended (~15-45 min with GPU)
- **VAE Generation**: GPU recommended (~20-60 min with GPU)
- **All Model Training**: GPU recommended (same as Phase 1)

**Estimated Total Timeline:**
- Classical only: 30 min (CPU)
- With GPU (all three): 3-4 hours
- With CPU (all three): 6-8 hours

## Prerequisites

- Phase 1 baseline complete, or raw `Dataset/my Dataset/` available
- TensorFlow, librosa, NumPy, scikit-learn
- Optional: GPU (highly recommended for Transformer and VAE)

## Documentation

- [Classical Augmentation Details](classical/README.md)
- [Transformer (MAE) Details](transformer/README.md)
- [VAE Details](vae_gan/README.md)
- [General Augmentation Guide](../AUGMENTATION_GUIDE.md)

## FAQ & Troubleshooting

**Q: Generation failed—what should I do?**
```bash
rm data/classical_augmented_features.npz
python classical/generate_classical_augmented_data.py
```

**Q: Models training is slow or not using GPU**
```python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
```

**Q: Memory error during VAE generation**
```bash
python vae_gan/generate_vae_augmented_data.py --batch_size 16 --num_generated_per_class 30
```

**Q: Which approach should I start with?**
→ Start with Classical (no GPU needed, fast), then try Transformer and VAE

---

**Next: Choose an approach and run the pipeline!**
