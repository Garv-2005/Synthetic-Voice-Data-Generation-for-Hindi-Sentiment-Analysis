# 🚀 Start Here: Quick Reference Guide

## First Time Setup

1. **Install Dependencies**:
   ```bash
   pip install numpy librosa tensorflow matplotlib seaborn scikit-learn tqdm joblib
   ```

2. **Verify GPU** (optional but recommended):
   ```python
   import tensorflow as tf
   print("GPUs:", tf.config.list_physical_devices('GPU'))
   ```

## 🎯 Quick Commands

### Run Complete Comparison (First Time)
```bash
cd research_pipeline/comparison
python run_comparison.py
```

### Resume Training (If Interrupted)
```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing
```

### Check What's Done
```bash
# See which models are complete
ls research_pipeline/results/

# Check if features exist
ls research_pipeline/data/baseline_features.npz
```

## 📊 What Gets Saved

| File | Location | Purpose |
|------|----------|---------|
| **Features** | `data/baseline_features.npz` | Extracted once, reused for all models |
| **Models** | `results/<model>/best_<model>_model.h5` | Trained model weights |
| **Results** | `results/<model>/<model>_results.json` | All metrics and training history |
| **Plots** | `results/<model>/*.png` | Visualizations |
| **Comparison** | `results/comparison/*` | Comparison charts and summary |

## 🔄 Resume Scenarios

### Scenario 1: Training Stopped After 3 Models
```bash
python run_comparison.py --skip_existing
```
→ Loads existing 3 models, trains remaining 3

### Scenario 2: Want to Retrain Everything
```bash
python run_comparison.py --force
```
→ Retrains all models from scratch

### Scenario 3: New Session, Continue Work
```bash
# 1. Check status
ls research_pipeline/results/

# 2. Resume
python run_comparison.py --skip_existing
```

## ⚡ GPU Status

The pipeline automatically detects GPU. You'll see:
- `✓ GPU detected: 1 GPU(s) available` → GPU will be used
- `⚠ No GPU detected` → Will use CPU (slower but works)

## 📚 Documentation

- **Full Guide**: `README.md`
- **Resume Guide**: `RESUME_GUIDE.md`
- **Quick Start**: `QUICK_START.md`
- **Comparison Details**: `comparison/README.md`

## 🎯 Most Common Commands

```bash
# First time
python run_comparison.py

# Resume
python run_comparison.py --skip_existing

# Quick test (few epochs)
python run_comparison.py --epochs 10

# Custom parameters
python run_comparison.py --epochs 150 --batch_size 64
```

---

**Ready to start?** → `cd research_pipeline/comparison && python run_comparison.py`
