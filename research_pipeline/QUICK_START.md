# Quick Start Guide

## 🎯 Running the Multi-Model Comparison (Recommended)

### Option 1: Complete Comparison Pipeline

Compare 6 different model architectures:

```bash
cd research_pipeline/comparison
python run_comparison.py
```

This will:
1. **Auto-detect and use GPU** (if available)
2. Extract features from your dataset (if needed)
3. Train 6 different models (CNN, LSTM, CNN-LSTM, ResNet, Transformer, SVM)
4. Generate comprehensive comparison reports
5. Create comparison visualizations

**Time**: 
- With GPU: ~1.5-2.5 hours
- With CPU: ~4-6 hours

### Option 2: Resume Training

If training was interrupted or you're continuing work:

```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing
```

This automatically skips already-trained models and continues with remaining ones.

## 🎯 Running the Baseline Pipeline

### Option 1: Complete Pipeline

Run everything at once:

```bash
cd research_pipeline/baseline
python run_baseline_pipeline.py
```

This will:
1. Extract features from your dataset
2. Train a baseline CNN model
3. Generate all visualizations and reports

**Time**: ~10-30 minutes with GPU, ~30-60 minutes with CPU

### Option 2: Step-by-Step

#### Step 1: Extract Features
```bash
cd research_pipeline/baseline
python extract_baseline_features.py
```

#### Step 2: Train Model
```bash
python train_baseline_model.py --data_path ../data/baseline_features.npz
```

## 📁 Expected Output Structure

```
research_pipeline/
├── data/
│   └── baseline_features.npz          # Extracted features
└── results/
    └── baseline/
        ├── baseline_results.json       # Evaluation metrics
        ├── dataset_analysis.json       # Dataset statistics
        ├── best_baseline_model.h5      # Trained model
        ├── dataset_distribution.png    # Distribution plot
        ├── sample_spectrograms.png     # Sample features
        ├── training_history.png        # Training curves
        ├── confusion_matrix.png        # Confusion matrix
        └── per_class_metrics.png       # Per-class performance
```

## 🔧 Customization

### Change Number of Epochs
```bash
python run_baseline_pipeline.py --epochs 150
```

### Change Batch Size
```bash
python run_baseline_pipeline.py --batch_size 64
```

### Change Random Seed
```bash
python run_baseline_pipeline.py --seed 123
```

## 📊 Viewing Results

### JSON Results
Open `results/baseline/baseline_results.json` for:
- Overall accuracy and F1-scores
- Per-class precision, recall, F1
- Training history
- Configuration details

### Visualizations
All plots are saved as high-resolution PNG files (300 DPI) in `results/baseline/`

### Model
Load the trained model:
```python
from tensorflow.keras.models import load_model
model = load_model('results/baseline/best_baseline_model.h5')
```

## ⚠️ Troubleshooting

### GPU Not Detected

If you have a GPU but it's not being used:
1. Verify CUDA and cuDNN are installed
2. Check TensorFlow GPU support:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```
3. Training will still work on CPU (slower)

### Import Errors
Make sure you're in the correct directory:
```bash
cd research_pipeline/baseline  # or comparison
```

### Dataset Not Found
Check that the dataset path is correct:
- Expected: `../../Dataset/my Dataset/`
- Adjust in `extract_baseline_features.py` if needed

### Out of Memory
Reduce batch size:
```bash
python run_comparison.py --batch_size 16
```

### Resume Training
If training was interrupted:
```bash
python run_comparison.py --skip_existing
```

## 📚 Next Steps

### After Multi-Model Comparison:
1. Review comparison results in `results/comparison/`
2. Identify best-performing model
3. Analyze per-model results in individual folders
4. Fine-tune the best model if needed

### After Baseline:
1. Review results in `results/baseline/`
2. Note which emotions are challenging
3. Proceed to multi-model comparison
4. Compare improvements over baseline

## 🔄 Resuming Work

### Quick Resume
```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing
```

### Check Status
```bash
# See which models are complete
ls research_pipeline/results/

# Check if features exist
ls research_pipeline/data/baseline_features.npz
```

See `RESUME_GUIDE.md` for detailed resume instructions.

---

**Ready?** 
- **Multi-model comparison**: `cd research_pipeline/comparison && python run_comparison.py`
- **Baseline only**: `cd research_pipeline/baseline && python run_baseline_pipeline.py`
