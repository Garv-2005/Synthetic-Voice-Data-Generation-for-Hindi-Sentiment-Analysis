# Multi-Model Comparison Guide

## 🎯 Overview

This guide explains how to use the multi-model comparison framework to train and compare different deep learning architectures for Hindi Speech Emotion Recognition.

## 🏗️ Architecture Overview

The framework implements 5 different model architectures, each with unique strengths:

| Model | Architecture | Best For | Key Features |
|-------|-------------|----------|--------------|
| **CNN** | Convolutional Neural Network | Spatial patterns | 4-layer CNN with batch norm |
| **LSTM** | Bidirectional LSTM | Temporal sequences | Long-term dependencies |
| **CNN-LSTM** | Hybrid CRNN | Both spatial & temporal | CNN + LSTM combination |
| **ResNet** | Residual Network | Deep architectures | Skip connections |
| **Transformer** | Multi-head Attention | Global context | Self-attention mechanism |

## 🚀 Quick Start

### Run Complete Comparison

```bash
cd research_pipeline/comparison
python run_comparison.py
```

This single command will:
1. Extract features (if needed)
2. Train all 5 models
3. Generate comprehensive comparisons
4. Save all results and visualizations

### Expected Output

```
results/
├── cnn/                    # CNN results
├── lstm/                   # LSTM results
├── cnn_lstm/              # CNN-LSTM results
├── resnet/                # ResNet results
├── transformer/           # Transformer results
└── comparison/            # Comparison visualizations
    ├── comparison_results.json
    ├── model_comparison.png
    ├── training_curves_comparison.png
    └── metrics_radar_chart.png
```

## 📊 Metrics Explained

### Overall Metrics

- **Accuracy**: Percentage of correct predictions
- **Macro F1**: Average F1 across all classes (unweighted)
- **Weighted F1**: Average F1 weighted by class frequency

### Per-Class Metrics

For each emotion class:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 📈 Visualizations

### Individual Model Visualizations

Each model generates:
1. **Training History**: Accuracy and loss over epochs
2. **Comprehensive Training History**: Includes F1 scores and learning rate
3. **Confusion Matrix**: Raw and normalized versions
4. **Per-Class Metrics**: Bar chart of precision, recall, F1

### Comparison Visualizations

1. **Model Comparison Bar Chart**
   - Side-by-side comparison of key metrics
   - Easy to identify best performing model

2. **Training Curves Comparison**
   - All models' training progress in one view
   - Compare convergence speed and stability

3. **Metrics Radar Chart**
   - Multi-dimensional comparison
   - Visual representation of strengths/weaknesses

## 🔧 Customization

### Train Specific Models Only

Edit `compare_models.py`:

```python
models_to_train = [
    (CNNModel, 'CNN'),
    (LSTMModel, 'LSTM'),
    # Comment out models you don't need
]
```

### Adjust Training Parameters

```bash
python run_comparison.py --epochs 150 --batch_size 64
```

### Train Individual Model

```python
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
```

## 📋 Results Interpretation

### Reading Results JSON

Each model's results include:

```json
{
  "model_name": "CNN",
  "metrics": {
    "overall": {
      "accuracy": 0.7234,
      "macro_f1": 0.6891,
      "weighted_f1": 0.7215
    },
    "per_class": {
      "anger": {
        "precision": 0.75,
        "recall": 0.70,
        "f1-score": 0.72
      }
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

### Comparison Results

The comparison JSON includes:
- Summary table of all models
- Best model identification
- Full results for each model

## ⚡ Performance Tips

1. **Start Small**: Test with fewer epochs first
   ```bash
   python run_comparison.py --epochs 20
   ```

2. **Monitor Resources**: Some models (Transformer, CNN-LSTM) use more memory

3. **Batch Size**: Reduce if you run out of memory
   ```bash
   python run_comparison.py --batch_size 16
   ```

4. **GPU Acceleration**: Models automatically use GPU if available

## 🎯 Best Practices

1. **Always compare on same data**: All models use the same train/val/test splits
2. **Check individual results**: Review each model's confusion matrix
3. **Look for patterns**: Which emotions are consistently difficult?
4. **Consider ensemble**: Best model might be a combination

## 🔍 Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 16`
- Train models one at a time

### Training Too Slow
- Reduce epochs for testing: `--epochs 20`
- Use fewer models initially

### Import Errors
- Make sure you're in the correct directory
- Check that all dependencies are installed

## 📚 Next Steps

After comparison:

1. **Analyze**: Which model performs best on your metrics?
2. **Investigate**: Why does one model outperform others?
3. **Optimize**: Fine-tune the best model's hyperparameters
4. **Ensemble**: Combine predictions from top models

## 📖 Related Documentation

- **Baseline Experiments**: `baseline/README.md`
- **Model Comparison Details**: `comparison/README.md`
- **Main Pipeline**: `README.md`
- **Quick Start**: `QUICK_START.md`

---

**Ready to compare?** Run: `python run_comparison.py`
