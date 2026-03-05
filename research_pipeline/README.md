# Research Pipeline: Synthetic Data Generation for Hindi Speech Emotion Recognition

This repository contains a well-organized, research-ready pipeline for implementing and evaluating synthetic data generation techniques for Hindi Speech Emotion Recognition (SER).

## 📁 Directory Structure

```
research_pipeline/
├── baseline/              # Baseline experiments (no augmentation)
│   ├── extract_baseline_features.py    # Feature extraction with analysis
│   ├── train_baseline_model.py         # Model training and evaluation
│   └── run_baseline_pipeline.py        # Complete baseline pipeline runner
├── models/                # Multiple model architectures
│   ├── cnn_model.py       # CNN baseline
│   ├── lstm_model.py     # Bidirectional LSTM
│   ├── cnn_lstm_model.py # CNN-LSTM hybrid (CRNN)
│   ├── resnet_model.py   # ResNet with residual connections
│   └── transformer_model.py # Transformer with attention
├── comparison/            # Model comparison framework
│   ├── train_model.py    # Generic model training script
│   ├── compare_models.py # Compare all models
│   └── run_comparison.py # Complete comparison pipeline
├── data/                  # Extracted features (generated)
│   └── baseline_features.npz
├── results/               # Experiment results and visualizations
│   ├── baseline/         # Baseline CNN results
│   ├── cnn/              # CNN model results
│   ├── lstm/             # LSTM model results
│   ├── cnn_lstm/         # CNN-LSTM model results
│   ├── resnet/           # ResNet model results
│   ├── transformer/      # Transformer model results
│   └── comparison/       # Comparison visualizations
└── utils/                 # Utility functions
    ├── __init__.py
    └── visualization.py   # Plotting and visualization functions
```

## 🚀 Quick Start

### Option 1: Multi-Model Comparison (Recommended)

Compare 6 different architectures on the same dataset:

```bash
cd research_pipeline/comparison
python run_comparison.py
```

This will:
1. **Detect and configure GPU** (if available) for faster training
2. Extract Mel-spectrogram features (if needed)
3. Train 6 different model architectures:
   - CNN (baseline)
   - LSTM (temporal modeling)
   - CNN-LSTM (hybrid)
   - ResNet (residual connections)
   - Transformer (attention mechanism)
   - SVM (classical ML baseline)
4. Generate comprehensive comparison reports
5. Create comparison visualizations

**Results**: Saved in `results/comparison/` with individual model results in separate folders.

### Resume Training

If training was interrupted or you're continuing in a new session:

```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing
```

This automatically skips already-trained models and continues with remaining ones.

### Option 2: Baseline Pipeline Only

Run just the baseline CNN model:

```bash
cd research_pipeline/baseline
python run_baseline_pipeline.py
```

This will:
1. Extract Mel-spectrogram features from the dataset
2. Analyze dataset distribution
3. Create train/validation/test splits
4. Train a baseline CNN model
5. Evaluate on test set
6. Generate comprehensive visualizations and reports

### View Results

**Multi-Model Comparison Results**:
- **Comparison**: `results/comparison/comparison_results.json`
- **Visualizations**: Model comparison charts, training curves, radar chart
- **Individual Models**: Each model has its own folder with detailed results

**Baseline Results**:
- **JSON Reports**: `results/baseline/baseline_results.json`, `dataset_analysis.json`
- **Visualizations**: Training curves, confusion matrices, per-class metrics
- **Model Weights**: `best_baseline_model.h5`

## 📊 Model Architectures

### Available Models

The pipeline includes 6 different architectures based on research literature:

#### 1. **CNN** (Baseline)
- **Input**: Mel-spectrograms (128 mel bands × 174 time frames)
- **Architecture**: 4 Conv Blocks (32 → 64 → 128 → 256 filters)
- **Features**: Batch Normalization, Dropout regularization
- **Best for**: Spatial pattern recognition in time-frequency domain

#### 2. **LSTM**
- **Architecture**: Bidirectional LSTM layers
- **Features**: Temporal sequence modeling
- **Best for**: Long-term temporal dependencies

#### 3. **CNN-LSTM** (CRNN)
- **Architecture**: CNN for spatial features + LSTM for temporal modeling
- **Features**: Hybrid approach combining both paradigms
- **Best for**: Both spatial and temporal feature learning

#### 4. **ResNet**
- **Architecture**: Residual connections with skip connections
- **Features**: Deeper networks with better gradient flow
- **Best for**: Deeper architectures with improved generalization

#### 5. **Transformer**
- **Architecture**: Multi-head self-attention mechanism
- **Features**: Global context modeling
- **Best for**: Capturing complex relationships across sequences

#### 6. **SVM** (Classical ML)
- **Architecture**: Support Vector Machine with RBF kernel
- **Features**: Classical machine learning approach
- **Best for**: Baseline comparison, interpretability, fast training

### Common Specifications

- **Input**: Mel-spectrograms (128 mel bands × 174 time frames)
- **Output**: 8 emotion classes (softmax)
- **Regularization**: Batch Normalization, Dropout (0.25-0.5)
- **Optimizer**: Adam
- **Loss**: Sparse categorical crossentropy

### Dataset

- **Source**: `Dataset/my Dataset/`
- **Emotions**: anger, disgust, fear, happy, neutral, sad, sarcastic, surprise
- **Speakers**: 8 speakers
- **Sessions**: 5 sessions per speaker
- **Split**: 70% train, 15% validation, 15% test (stratified)

### Feature Extraction

- **Mel-spectrograms**: 128 mel filter banks
- **Time frames**: 174 (padded/truncated)
- **Sample rate**: 16 kHz
- **Preprocessing**: Trim silences, normalize

## 📝 Running Individual Components

### Extract Features Only

```bash
cd research_pipeline/baseline
python extract_baseline_features.py
```

### Train Model Only

```bash
cd research_pipeline/baseline
python train_baseline_model.py --data_path ../data/baseline_features.npz
```

## 📈 Generated Documentation

### 1. Dataset Analysis (`dataset_analysis.json`)
- Total samples
- Distribution across emotion classes
- Class balance statistics

### 2. Baseline Results (`baseline_results.json`)
- Overall metrics (accuracy, F1-scores)
- Per-class metrics (precision, recall, F1)
- Training history
- Configuration details

### 3. Visualizations

**Dataset Distribution** (`dataset_distribution.png`)
- Bar chart showing samples per emotion class
- Percentage breakdown

**Sample Spectrograms** (`sample_spectrograms.png`)
- Example Mel-spectrograms for each emotion

**Training History** (`training_history.png`)
- Training/validation accuracy curves
- Training/validation loss curves

**Confusion Matrix** (`confusion_matrix.png` & `confusion_matrix_normalized.png`)
- Classification performance matrix
- Both raw counts and normalized versions

**Per-Class Metrics** (`per_class_metrics.png`)
- Precision, recall, and F1-score for each emotion class

### 4. Comparison Visualizations (Multi-Model)

**Model Comparison** (`comparison/model_comparison.png`)
- Side-by-side bar charts for accuracy, macro F1, weighted F1

**Training Curves Comparison** (`comparison/training_curves_comparison.png`)
- All models' training/validation curves in one view

**Metrics Radar Chart** (`comparison/metrics_radar_chart.png`)
- Multi-dimensional comparison across all metrics

**Comprehensive Training History** (per model)
- Includes F1 score curves and learning rate tracking

## 🔧 Configuration

### Adjust Training Parameters

Edit `run_baseline_pipeline.py` or use command-line arguments:

```bash
python run_baseline_pipeline.py --epochs 150 --batch_size 64
```

### Adjust Model Architecture

Edit `train_baseline_model.py` → `BaselineModel.build_model()`

### Adjust Feature Extraction

Edit `extract_baseline_features.py` → `BaselineFeatureExtractor.__init__()`

## 📋 Requirements

```
numpy>=1.21.0
librosa>=0.9.0
tensorflow>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.64.0
joblib>=1.1.0
```

## ⚙️ Setup (Fresh Dev Machine)

Follow these steps on a fresh machine to create a virtual environment and install Python dependencies.

- Windows (PowerShell):

```powershell
cd research_pipeline
.\install_deps.ps1 -venvName myenv
```

- macOS / Linux (bash):

```bash
cd research_pipeline
./install_deps.sh myenv
```

Notes:
- The scripts create a virtual environment, upgrade `pip`, and install packages from `requirements.txt`.
- GPU drivers, CUDA and cuDNN must be installed manually following NVIDIA's instructions if you need TensorFlow GPU support. See `TENSORFLOW_GPU_CONFIG.md` for guidance.

## ▶️ Quick Test After Setup

After installing dependencies, run the small diagnostics included in `tests/` to verify installation and (if available) GPU accessibility:

```bash
cd research_pipeline
python -m pip install -r requirements.txt
python tests/gpu_diagnostic.py
python tests/test_augmentation.py
```

The diagnostic scripts will report TensorFlow version, whether it was built with CUDA support, and run a tiny training job to validate model builds.

### GPU Support (Optional but Recommended)

For faster training, install CUDA and cuDNN:
- **CUDA**: https://developer.nvidia.com/cuda-downloads
- **cuDNN**: https://developer.nvidia.com/cudnn

The pipeline automatically detects and uses GPU if available. No additional configuration needed!

## 🎯 Next Steps

### After Model Comparison:

1. **Analyze Results**: Review `results/comparison/comparison_results.json`
2. **Identify Best Model**: Check which architecture performs best
3. **Fine-tune Best Model**: Further optimize hyperparameters
4. **Ensemble Methods**: Combine predictions from multiple models

### Synthetic Data Augmentation (Phase 2):

Three approaches are implemented under `augmentation/` (see **AUGMENTATION_GUIDE.md**):

1. **Classical** (`augmentation/classical/`): Time stretch, pitch shift, noise, volume on audio → same Mel extraction and 6-model evaluation.
2. **Transformer (MAE)** (`augmentation/transformer/`): Masked autoencoder on spectrograms; generate new spectrograms and evaluate with same models.
3. **VAE** (`augmentation/vae_gan/`): VAE on spectrograms; sample and decode to new spectrograms; same evaluation.

Results for each approach are under `results/augmentation/<approach>/` so you can compare baseline vs augmentation using the same metrics.

## 📚 Research Paper Integration

All results are formatted for research paper integration:
- JSON results can be imported into analysis scripts
- High-resolution figures (300 DPI) ready for publication
- Comprehensive metrics for statistical analysis
- Reproducible experiments with fixed random seeds

## 🤝 Citation

If using this pipeline for research, please cite the base paper:
> Design and Validation of HindiSER: Speech Emotion Recognition Dataset for Hindi Language

## 📞 Support

For issues or questions, please check:
1. Ensure dataset path is correct (`../../Dataset/my Dataset/`)
2. Check all dependencies are installed
3. Verify sufficient disk space for generated files

## 📖 Additional Documentation

- **Baseline Experiments**: See `baseline/README.md`
- **Model Comparison**: See `comparison/README.md`
- **Synthetic Data Augmentation**: See `AUGMENTATION_GUIDE.md` and `augmentation/README.md`
- **Quick Start Guide**: See `QUICK_START.md`
- **Resume Guide**: See `RESUME_GUIDE.md` (How to continue from interruptions)
- **Dataset Information**: See `../Dataset/README.md`

## 🔄 Resuming Work

### Quick Resume Command

```bash
cd research_pipeline/comparison
python run_comparison.py --skip_existing
```

### What Gets Saved (Can Resume From)

1. **Features**: `data/baseline_features.npz` - Extracted once, reused
2. **Trained Models**: `results/<model>/best_<model>_model.h5` - Saved after each model
3. **Results**: `results/<model>/<model>_results.json` - All metrics and history
4. **Visualizations**: `results/<model>/*.png` - Generated plots

### Check What's Done

```bash
# Check which models are complete
ls research_pipeline/results/

# Check if features exist
ls research_pipeline/data/baseline_features.npz
```

See `RESUME_GUIDE.md` for detailed instructions on resuming work.

---

**Status**: 
- ✅ Baseline pipeline complete
- ✅ Multi-model comparison framework ready
- ✅ 6 model architectures implemented (including SVM)
- ✅ Comprehensive comparison visualizations
- ✅ GPU support with automatic detection
- ✅ Resume functionality (skip existing models)
- ✅ Progress bars for all operations

**Augmentation**: Classical, Transformer (MAE), and VAE pipelines in `augmentation/` (see `AUGMENTATION_GUIDE.md`).
