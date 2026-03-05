# Resume Guide: Picking Up Where You Left Off

This guide helps you quickly resume work on the multi-model comparison pipeline after an interruption or in a new session.

## 🚀 Quick Resume

### Resume Training (Skip Already-Trained Models)

If you've already trained some models and want to continue with the rest:

```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing
```

This will:
- ✅ Skip models that are already trained
- ✅ Train only models that haven't been completed
- ✅ Load existing results for comparison

### Check What's Already Done

Before resuming, check which models are complete:

```bash
# Check results directory
ls research_pipeline/results/

# Check individual model results
ls research_pipeline/results/cnn/
ls research_pipeline/results/lstm/
# etc.
```

A model is considered complete if both exist:
- `best_<model>_model.h5` (or `.pkl` for SVM)
- `<model>_results.json`

## 📋 Status Checking

### Check Feature Extraction Status

```bash
# Check if features exist
ls research_pipeline/data/baseline_features.npz
```

If this file exists, feature extraction will be skipped automatically.

### Check Model Training Status

Each model's status can be checked by looking for:
- **Model file**: `results/<model_name>/best_<model>_model.h5` (or `.pkl` for SVM)
- **Results file**: `results/<model_name>/<model>_results.json`

### Quick Status Script

Create a simple status checker:

```python
import os
from pathlib import Path

results_dir = Path("research_pipeline/results")
models = ['cnn', 'lstm', 'cnn_lstm', 'resnet', 'transformer', 'svm']

print("Model Training Status:")
print("=" * 50)
for model in models:
    model_dir = results_dir / model
    model_file = model_dir / f"best_{model}_model.h5" if model != 'svm' else model_dir / f"best_{model}_model.pkl"
    results_file = model_dir / f"{model}_results.json"
    
    if model_file.exists() and results_file.exists():
        print(f"✓ {model.upper():12s} - Complete")
    elif model_dir.exists():
        print(f"⚠ {model.upper():12s} - Incomplete")
    else:
        print(f"✗ {model.upper():12s} - Not started")
```

## 🔄 Resume Scenarios

### Scenario 1: Training Interrupted Midway

**Situation**: Training stopped after 3 models (e.g., CNN, LSTM, CNN-LSTM completed)

**Solution**:
```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing
```

This will:
- Load results from CNN, LSTM, CNN-LSTM
- Train remaining models (ResNet, Transformer, SVM)
- Generate comparison with all models

### Scenario 2: Want to Retrain Specific Model

**Situation**: Need to retrain one model with different parameters

**Solution**:
```bash
# Train individual model
cd research_pipeline/comparison
python -c "
from train_model import train_model
from models import CNNModel

train_model(
    model_class=CNNModel,
    data_path='../data/baseline_features.npz',
    results_dir='../results/cnn',
    model_name='CNN',
    epochs=150,  # Different epochs
    batch_size=64  # Different batch size
)
"
```

### Scenario 3: Force Retrain All Models

**Situation**: Want to retrain everything from scratch

**Solution**:
```bash
cd research_pipeline/comparison
python run_comparison.py --force
```

This will retrain all models even if they exist.

### Scenario 4: New Session - Continue Previous Work

**Situation**: Starting fresh session, want to continue where you left off

**Steps**:
1. **Check status** (see above)
2. **Resume training**:
   ```bash
   cd research_pipeline/comparison
   python run_comparison.py --skip_existing
   ```
3. **Verify GPU is available** (should see GPU message at start)

## 💾 What Gets Saved

### Automatically Saved (Can Resume From)

1. **Features**: `research_pipeline/data/baseline_features.npz`
   - Extracted once, reused for all models
   - Skip extraction if this exists

2. **Trained Models**: `results/<model>/best_<model>_model.h5`
   - Saved after each model completes
   - Can load and use for inference

3. **Results JSON**: `results/<model>/<model>_results.json`
   - Contains all metrics and training history
   - Used for comparison generation

4. **Visualizations**: `results/<model>/*.png`
   - Generated after each model
   - Can regenerate if needed

### Comparison Results

- `results/comparison/comparison_results.json` - Summary of all models
- `results/comparison/*.png` - Comparison visualizations

## 🔧 Command-Line Options

### Resume Options

```bash
# Skip already-trained models
python run_comparison.py --skip_existing

# Force retrain everything
python run_comparison.py --force

# Skip features if they exist
python run_comparison.py --skip_features

# Combine options
python run_comparison.py --skip_existing --epochs 150 --batch_size 64
```

### Full Option List

```bash
python run_comparison.py \
    --epochs 100 \
    --batch_size 32 \
    --skip_existing \
    --force \
    --skip_features
```

## 🎯 Workflow Examples

### Example 1: First Time Run

```bash
cd research_pipeline/comparison
python run_comparison.py
```

### Example 2: Resume After Interruption

```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing
```

### Example 3: Test Run (Few Epochs)

```bash
cd research_pipeline/comparison
python run_comparison.py --epochs 10
```

### Example 4: Production Run (Resume if Needed)

```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing --epochs 150
```

## 📊 Understanding Results

### Check Individual Model Results

```bash
# View results JSON
cat research_pipeline/results/cnn/cnn_results.json | python -m json.tool
```

### Check Comparison Summary

```bash
# View comparison results
cat research_pipeline/results/comparison/comparison_results.json | python -m json.tool
```

## ⚡ GPU Configuration

The pipeline automatically detects and uses GPU if available. You'll see:

```
✓ GPU detected: 1 GPU(s) available
  Using GPU: /physical_device:GPU:0
```

If no GPU is detected:
```
⚠ No GPU detected. Training will use CPU (slower).
```

### Verify GPU Setup

```python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
```

## 🐛 Troubleshooting Resume

### Issue: Model exists but results.json missing

**Solution**: Delete the model file and retrain:
```bash
rm research_pipeline/results/cnn/best_cnn_model.h5
python run_comparison.py --skip_existing
```

### Issue: Want to retrain with different parameters

**Solution**: Use `--force` flag:
```bash
python run_comparison.py --force --epochs 150
```

### Issue: Features corrupted or need regeneration

**Solution**: Delete features and regenerate:
```bash
rm research_pipeline/data/baseline_features.npz
python run_comparison.py
```

## 📝 Best Practices

1. **Always use `--skip_existing`** when resuming to avoid retraining
2. **Check status first** before running full pipeline
3. **Save comparison results** - they're regenerated each run
4. **Keep feature file** - it's reused for all models
5. **Monitor GPU usage** - ensure GPU is being used for faster training

## 🔗 Related Documentation

- **Main README**: `research_pipeline/README.md`
- **Comparison Guide**: `research_pipeline/comparison/README.md`
- **Quick Start**: `research_pipeline/QUICK_START.md`

---

**Quick Command Reference**:
- Resume: `python run_comparison.py --skip_existing`
- Force retrain: `python run_comparison.py --force`
- Check status: `ls research_pipeline/results/`
