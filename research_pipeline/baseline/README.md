# Baseline Experiments

This directory contains scripts for establishing a baseline model performance on the original (non-augmented) dataset.

## Scripts

### 1. `extract_baseline_features.py`
Extracts Mel-spectrogram features from the original dataset.

**Features:**
- Mel-spectrogram extraction (128 mel bands, 174 time frames)
- Dataset analysis and statistics
- Train/validation/test split (70/15/15)
- Automatic visualization generation

**Output:**
- `../data/baseline_features.npz` - Extracted features
- `../results/baseline/dataset_analysis.json` - Dataset statistics
- `../results/baseline/dataset_distribution.png` - Distribution plot
- `../results/baseline/sample_spectrograms.png` - Sample visualizations

**Usage:**
```bash
python extract_baseline_features.py
```

### 2. `train_baseline_model.py`
Trains a CNN model on baseline features.

**Features:**
- 4-layer CNN with batch normalization
- Dropout for regularization
- Early stopping and learning rate reduction
- Comprehensive evaluation metrics
- Automatic visualization generation

**Output:**
- `../results/baseline/best_baseline_model.h5` - Trained model
- `../results/baseline/baseline_results.json` - Results and metrics
- `../results/baseline/training_history.png` - Training curves
- `../results/baseline/confusion_matrix.png` - Confusion matrix
- `../results/baseline/per_class_metrics.png` - Per-class performance

**Usage:**
```bash
python train_baseline_model.py --data_path ../data/baseline_features.npz
```

### 3. `run_baseline_pipeline.py`
Runs the complete baseline pipeline end-to-end.

**Usage:**
```bash
python run_baseline_pipeline.py
```

**With options:**
```bash
python run_baseline_pipeline.py --epochs 150 --batch_size 64 --seed 42
```

## Expected Results

Based on typical Hindi SER datasets:
- **Test Accuracy**: ~65-75%
- **Macro F1-Score**: ~0.60-0.70
- **Best performing emotions**: Usually happy, sad, neutral
- **Challenging emotions**: Often sarcastic, disgust

## Notes

- All experiments use fixed random seed (42) for reproducibility
- Stratified splitting ensures balanced class distribution
- Model checkpoints save best model based on validation loss
- Results are saved in JSON format for easy analysis
- **GPU is automatically detected and used** if available (faster training)
- Features are saved and can be reused for other experiments

## 🔄 Resuming Work

If you need to retrain or continue work:
- Features are saved in `../data/baseline_features.npz` (reused automatically)
- Model is saved in `../results/baseline/best_baseline_model.h5`
- Results are saved in `../results/baseline/baseline_results.json`

To retrain with different parameters, simply run the pipeline again - it will overwrite previous results.
