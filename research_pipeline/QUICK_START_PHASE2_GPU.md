# Quick Start: Running Phase 2 on Faster GPU System

This guide is for running the Phase 2 augmentation pipeline on your faster GPU machine after pushing to GitHub.

## Prerequisites

```bash
# Clone repository
git clone <your-repo-url>
cd Capstone/research_pipeline

# Install dependencies
pip install numpy librosa tensorflow matplotlib seaborn scikit-learn tqdm joblib scikit-learn

# Verify GPU setup
python tests/gpu_diagnostic.py
```

## Three Approaches: Quick Reference

| Approach | Time | Hardware | Commands |
|----------|------|----------|----------|
| **Classical** | ~2.5h | CPU+GPU | `python augmentation/classical/run_classical_pipeline.py --epochs 100` |
| **VAE** | ~2.5h | GPU | `python augmentation/vae_gan/generate_vae_augmented_data.py --vae_epochs 50` + `python augmentation/vae_gan/train_vae_models.py --epochs 100` |
| **Transformer** | ~3h | GPU | `python augmentation/transformer/generate_transformer_augmented_data.py --mae_epochs 30` + `python augmentation/transformer/train_transformer_models.py --epochs 100` |

## Running Individual Approaches

### Option 1: Classical Augmentation (Fastest)

```bash
cd research_pipeline/augmentation/classical

# Full pipeline (generation + training)
python run_classical_pipeline.py --epochs 100

# Or separate stages
python generate_classical_augmented_data.py        # ~10 min
python train_classical_models.py --epochs 100     # ~2 hours
```

**Output:**
- Features: `data/classical_augmented_features.npz`
- Results: `results/augmentation/classical/comparison/comparison_results.json`

### Option 2: VAE Augmentation (Best Balance)

```bash
cd research_pipeline/augmentation/vae_gan

# Generation (train VAE, generate spectrograms)
python generate_vae_augmented_data.py --vae_epochs 50  # ~60 min

# Training (train 6 models on augmented data)
python train_vae_models.py --epochs 100                # ~2 hours

# Or full pipeline
python run_vae_pipeline.py --vae_epochs 50 --epochs 100
```

**Output:**
- Features: `data/vae_augmented_features.npz`
- Results: `results/augmentation/vae/comparison/comparison_results.json`

### Option 3: Transformer (MAE) - Best Results

```bash
cd research_pipeline/augmentation/transformer

# Generation (train MAE, generate spectrograms)
python generate_transformer_augmented_data.py --mae_epochs 30  # ~60 min

# Training (train 6 models on augmented data)
python train_transformer_models.py --epochs 100                # ~2 hours

# Or full pipeline
python run_transformer_pipeline.py --mae_epochs 30 --epochs 100
```

**Output:**
- Features: `data/transformer_augmented_features.npz`
- Results: `results/augmentation/transformer/comparison/comparison_results.json`

## Running All Three Approaches (Recommended for Comparison)

```bash
cd research_pipeline

# Run all three approaches sequentially
python augmentation/classical/run_classical_pipeline.py --epochs 100

python augmentation/vae_gan/generate_vae_augmented_data.py --vae_epochs 50
python augmentation/vae_gan/train_vae_models.py --epochs 100

python augmentation/transformer/generate_transformer_augmented_data.py --mae_epochs 30
python augmentation/transformer/train_transformer_models.py --epochs 100

# Total time: ~8 hours on good GPU
```

## Comparing Results

After running all approaches, compare results:

```bash
# Read summary files
# Baseline (Phase 1):
results/comparison/comparison_results.json

# Phase 2 approaches:
results/augmentation/classical/comparison/comparison_results.json
results/augmentation/vae/comparison/comparison_results.json
results/augmentation/transformer/comparison/comparison_results.json
```

### Expected Results (Approximate)

| Approach | Training Data | Model | Accuracy | Macro F1 |
|----------|-----------------|--------|-----------|----------|
| **Baseline** | 2,240 | CNN | ~70% | ~70% |
| **Classical** | ~11,200 | CNN | ~72-74% | ~72-74% |
| **VAE** | ~11,200 | Transformer | ~76-78% | ~76-78% |
| **Transformer** | ~11,200 | Transformer | ~77-79% | ~77-79% |

## Customization Options

### Epochs and Batch Size
```bash
python run_classical_pipeline.py --epochs 150 --batch_size 64
```

### GPU Memory Issues
If you run out of GPU memory, reduce batch size:
```bash
python generate_vae_augmented_data.py --batch_size 16  # Default 32
```

### Resume from Interruption
```bash
# Skip already-trained models
python train_vae_models.py --skip_existing

# Force retrain all
python train_vae_models.py --force
```

## Troubleshooting

### GPU not detected
```bash
python tests/gpu_diagnostic.py  # Check TensorFlow GPU setup
nvidia-smi                      # Check NVIDIA driver
```

### Out of memory
- Reduce batch size: `--batch_size 16`
- Reduce epochs: `--epochs 50`
- Generate data separately from training

### Import errors
```bash
pip install --upgrade tensorflow scikit-learn librosa
```

## Understanding the Results

Each approach generates a `comparison_results.json` with:
- **accuracy**: Overall classification accuracy
- **macro_f1**: Unweighted F1 score (treats all emotions equally)
- **weighted_f1**: Weighted by class frequency
- **per_class**: Precision, recall, F1 for each emotion

Best metric for this project: **Macro F1** (all emotions equally important)

## Next Steps

1. ✓ Run diagnostics: `python tests/gpu_diagnostic.py`
2. ✓ Example test: `python tests/test_augmentation.py`
3. → Run Classical approach (quickest validation)
4. → Run VAE or Transformer (for presentation results)
5. → Compare all three against baseline
6. → Document findings in presentation

## Key Files

| File | Purpose |
|------|---------|
| `tests/gpu_diagnostic.py` | Diagnose GPU/TensorFlow issues |
| `tests/test_augmentation.py` | Quick tests before full training |
| `PHASE2_APPROACHES_ANALYSIS.md` | Detailed analysis of all approaches |
| `augmentation/README.md` | Master augmentation guide |
| `augmentation/[approach]/README.md` | Approach-specific details |
