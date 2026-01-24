# Model Comparison Framework

This directory contains scripts for training and comparing multiple deep learning architectures for Hindi Speech Emotion Recognition.

## 📋 Overview

The comparison framework trains 6 different model architectures on the same dataset and generates comprehensive comparison reports and visualizations.

## 🏗️ Model Architectures

### 1. **CNN** (Baseline)
- 4-layer convolutional neural network
- Spatial feature extraction from mel-spectrograms
- Best for: Capturing local patterns in time-frequency domain

### 2. **LSTM**
- Bidirectional LSTM for temporal sequence modeling
- Processes mel-spectrograms as time sequences
- Best for: Capturing long-term temporal dependencies

### 3. **CNN-LSTM** (CRNN)
- Hybrid architecture combining CNN and LSTM
- CNN extracts spatial features, LSTM models temporal patterns
- Best for: Both spatial and temporal feature learning

### 4. **ResNet**
- Residual connections for deeper networks
- Skip connections help with gradient flow
- Best for: Deeper architectures with better generalization

### 5. **Transformer**
- Multi-head attention mechanism
- Self-attention for sequence modeling
- Best for: Capturing global dependencies and relationships

### 6. **SVM** (Classical ML)
- Support Vector Machine with RBF kernel
- Classical machine learning baseline
- Best for: Comparison with deep learning approaches, interpretability

## 🚀 Quick Start

### Run Complete Comparison

Train all models and generate comparison:

```bash
cd research_pipeline/comparison
python run_comparison.py
```

With custom parameters:

```bash
python run_comparison.py --epochs 150 --batch_size 64
```

### Resume Training (Skip Already-Trained Models)

If training was interrupted or you want to continue from where you left off:

```bash
python run_comparison.py --skip_existing
```

This will:
- ✅ Skip models that are already trained
- ✅ Load existing results for comparison
- ✅ Train only remaining models

### Force Retrain

To retrain all models even if they exist:

```bash
python run_comparison.py --force
```

### Train Individual Model

Train a specific model:

```bash
python -c "
from comparison.train_model import train_model
from models import CNNModel

train_model(
    model_class=CNNModel,
    data_path='../data/baseline_features.npz',
    results_dir='../results/cnn',
    model_name='CNN',
    epochs=100,
    batch_size=32
)
"
```

### Compare All Models

If models are already trained, generate comparison:

```bash
python compare_models.py --data_path ../data/baseline_features.npz
```

## 📊 Output Structure

```
results/
├── cnn/                          # CNN model results
│   ├── best_cnn_model.h5
│   ├── cnn_results.json
│   ├── training_history.png
│   ├── comprehensive_training_history.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   └── per_class_metrics.png
├── lstm/                         # LSTM model results
│   └── ...
├── cnn_lstm/                     # CNN-LSTM model results
│   └── ...
├── resnet/                       # ResNet model results
│   └── ...
├── transformer/                  # Transformer model results
│   └── ...
├── svm/                         # SVM model results
│   └── ...
└── comparison/                   # Comparison visualizations
    ├── comparison_results.json
    ├── model_comparison.png
    ├── training_curves_comparison.png
    └── metrics_radar_chart.png
```

## 📈 Generated Visualizations

### Individual Model Results

Each model generates:
1. **Training History**: Accuracy and loss curves
2. **Comprehensive Training History**: Includes F1 scores and learning rate
3. **Confusion Matrix**: Raw and normalized
4. **Per-Class Metrics**: Precision, recall, F1 for each emotion

### Comparison Visualizations

1. **Model Comparison Bar Chart** (`model_comparison.png`)
   - Side-by-side comparison of accuracy, macro F1, weighted F1

2. **Training Curves Comparison** (`training_curves_comparison.png`)
   - All models' training/validation curves in one view

3. **Metrics Radar Chart** (`metrics_radar_chart.png`)
   - Multi-dimensional comparison of all metrics

## 📝 Results JSON Format

Each model's results JSON includes:

```json
{
  "model_name": "CNN",
  "timestamp": "2026-01-23T...",
  "configuration": {
    "input_shape": [128, 174, 1],
    "num_classes": 8,
    "epochs": 100,
    "batch_size": 32,
    "random_seed": 42
  },
  "metrics": {
    "overall": {
      "accuracy": 0.7234,
      "macro_f1": 0.6891,
      "weighted_f1": 0.7215
    },
    "per_class": {
      "anger": {"precision": 0.75, "recall": 0.70, "f1-score": 0.72},
      ...
    }
  },
  "training_history": {
    "accuracy": [...],
    "val_accuracy": [...],
    "loss": [...],
    "val_loss": [...],
    "f1": [...],
    "val_f1": [...]
  }
}
```

## 🔧 Configuration

### Training Parameters

Edit `compare_models.py` or use command-line arguments:

```bash
python compare_models.py \
    --epochs 150 \
    --batch_size 64 \
    --seed 42
```

### Model Selection

To train only specific models, edit `compare_models.py`:

```python
models_to_train = [
    (CNNModel, 'CNN'),
    (LSTMModel, 'LSTM'),
    # Comment out models you don't want to train
]
```

## 📊 Metrics Explained

### Overall Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1**: Average F1-score across all classes (unweighted)
- **Weighted F1**: Average F1-score weighted by class support

### Per-Class Metrics

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## ⏱️ Expected Training Times

Approximate training times:

**With GPU** (recommended):
- **CNN**: ~10-15 minutes
- **LSTM**: ~15-20 minutes
- **CNN-LSTM**: ~20-30 minutes
- **ResNet**: ~12-18 minutes
- **Transformer**: ~18-25 minutes
- **SVM**: ~5-10 minutes (fastest, no iterative training)

**With CPU** (slower):
- **CNN**: ~30-45 minutes
- **LSTM**: ~45-60 minutes
- **CNN-LSTM**: ~60-90 minutes
- **ResNet**: ~40-55 minutes
- **Transformer**: ~50-70 minutes
- **SVM**: ~5-10 minutes

**Total with GPU**: ~1.5-2.5 hours for complete comparison
**Total with CPU**: ~4-6 hours for complete comparison

### GPU Detection

The pipeline automatically detects and uses GPU if available. You'll see:
```
✓ GPU detected: 1 GPU(s) available
  Using GPU: /physical_device:GPU:0
```

If no GPU is detected, training will use CPU (slower but still works).

## 🎯 Best Practices

1. **Start with fewer epochs** for initial testing:
   ```bash
   python run_comparison.py --epochs 20
   ```

2. **Use GPU for faster training** - GPU is automatically detected and used

3. **Resume training** if interrupted:
   ```bash
   python run_comparison.py --skip_existing
   ```

4. **Check individual results** before running full comparison

5. **Save intermediate results** - models are saved after each training

6. **Monitor progress** - Use progress bars to track training status

## 🔍 Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python compare_models.py --batch_size 16
```

### Training Too Slow

Reduce epochs for initial testing:
```bash
python compare_models.py --epochs 50
```

### Import Errors

Make sure you're in the correct directory:
```bash
cd research_pipeline/comparison
```

## 📚 Next Steps

After comparison:

1. **Analyze Results**: Review `comparison/comparison_results.json`
2. **Identify Best Model**: Check which model performs best on your metrics
3. **Fine-tune**: Further optimize the best-performing model
4. **Ensemble**: Consider combining predictions from multiple models

## 🔄 Resuming Work

### Quick Resume

If training was interrupted or you're starting a new session:

```bash
python run_comparison.py --skip_existing
```

This automatically:
- Checks which models are already trained
- Loads existing results
- Trains only remaining models
- Generates comparison with all available results

### Check Status

See which models are complete:
```bash
ls research_pipeline/results/
```

A model is complete if both exist:
- `best_<model>_model.h5` (or `.pkl` for SVM)
- `<model>_results.json`

### Full Resume Guide

See `../RESUME_GUIDE.md` for detailed instructions on:
- Checking training status
- Resuming from interruptions
- Retraining specific models
- Understanding saved files

---

**Status**: Multi-model comparison framework ready ✓
**Models**: 6 architectures implemented (including SVM)
**Visualizations**: Comprehensive comparison charts included
**GPU Support**: Automatic GPU detection and usage
**Resume Support**: Skip existing models with `--skip_existing`
