# Hindi Speech Emotion Recognition Dataset

This directory contains the audio datasets used for Hindi Speech Emotion Recognition (SER) research.

## 📁 Directory Structure

```
Dataset/
├── my Dataset/                    # Primary dataset (used in baseline experiments)
│   ├── anger/                     # Anger emotion samples
│   ├── disgust/                   # Disgust emotion samples
│   ├── fear/                      # Fear emotion samples
│   ├── happy/                     # Happy emotion samples
│   ├── neutral/                   # Neutral emotion samples
│   ├── sad/                       # Sad emotion samples
│   ├── sarcastic/                 # Sarcastic emotion samples
│   └── surprise/                  # Surprise emotion samples
│
└── SER HINDI SESSION 1/          # Alternative dataset structure (by speaker/session)
    ├── 1/                         # Speaker 1
    │   ├── session1/              # Session 1 recordings
    │   ├── session2/              # Session 2 recordings
    │   └── session3/              # Session 3 recordings (organized by emotion)
    ├── 2/                         # Speaker 2
    ├── 3/                         # Speaker 3
    └── ...                        # Additional speakers
```

## 🎯 Primary Dataset: `my Dataset/`

### Overview

The primary dataset is organized by emotion classes. Each emotion folder contains audio samples in WAV format.

### Emotion Classes

The dataset includes **8 emotion classes**:

1. **anger** - 400 samples
2. **disgust** - 400 samples
3. **fear** - 400 samples
4. **happy** - 400 samples
5. **neutral** - 400 samples
6. **sad** - 400 samples
7. **sarcastic** - 400 samples
8. **surprise** - 400 samples

**Total**: 3,200 audio samples (balanced across all classes)

### Dataset Statistics

- **Total Samples**: 3,200
- **Number of Classes**: 8
- **Class Balance**: Perfectly balanced (400 samples per class)
- **Format**: WAV audio files
- **Language**: Hindi

### Usage in Research Pipeline

This dataset is used as the primary source for baseline experiments:

- **Path in pipeline**: `../../Dataset/my Dataset/`
- **Feature extraction**: Mel-spectrograms (128 mel bands, 174 time frames)
- **Sample rate**: 16 kHz
- **Train/Val/Test Split**: 70% / 15% / 15% (stratified)

## 📊 Alternative Dataset: `SER HINDI SESSION 1/`

### Overview

This dataset is organized by speaker and session, providing additional metadata about recording conditions.

### Structure

- **Speakers**: Multiple speakers (numbered 1, 2, 3, etc.)
- **Sessions**: Multiple recording sessions per speaker (session1, session2, session3)
- **Emotions**: Same 8 emotion classes as primary dataset
- **Organization**: Some sessions organized by emotion subfolders

### Use Cases

This dataset structure is useful for:
- Speaker-dependent analysis
- Session-based evaluation
- Cross-session validation
- Speaker adaptation experiments

## 🔧 Dataset Processing

### Feature Extraction

The research pipeline extracts features from audio files:

1. **Preprocessing**:
   - Load audio at 16 kHz sample rate
   - Trim leading/trailing silence
   - Normalize audio

2. **Feature Extraction**:
   - Mel-spectrogram computation
   - 128 mel filter banks
   - Fixed length: 174 time frames (padding/truncation)

3. **Output Format**:
   - Features saved as `.npz` files
   - Train/validation/test splits included
   - Emotion mapping preserved

### Example Usage

```python
from pathlib import Path
import librosa

# Load audio file
audio_path = Path("Dataset/my Dataset/anger/sample.wav")
y, sr = librosa.load(audio_path, sr=16000)

# Extract mel-spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=y, 
    sr=sr, 
    n_mels=128,
    hop_length=512,
    n_fft=2048
)
```

## 📈 Dataset Analysis

The research pipeline automatically generates dataset analysis:

- **Distribution plots**: Visual representation of class balance
- **Statistics**: JSON files with detailed metrics
- **Sample visualizations**: Example spectrograms per emotion

See `research_pipeline/results/baseline/dataset_analysis.json` for detailed statistics.

## 🎤 Recording Details

### Audio Specifications

- **Format**: WAV (uncompressed)
- **Sample Rate**: Variable (resampled to 16 kHz in pipeline)
- **Channels**: Mono (typically)
- **Duration**: Variable (processed to fixed-length features)

### Recording Conditions

- **Language**: Hindi
- **Emotions**: Acted/elicited emotions
- **Speakers**: Multiple native Hindi speakers
- **Environment**: Controlled recording conditions

## 📝 Notes

1. **File Naming**: Audio files may follow different naming conventions:
   - Primary dataset: Organized by emotion folder
   - Alternative dataset: May include speaker/session identifiers in filenames

2. **Data Quality**: 
   - All files are validated during feature extraction
   - Corrupted or unreadable files are skipped with error messages

3. **Storage**: 
   - Dataset contains ~3,200+ audio files
   - Ensure sufficient disk space for processing
   - Feature extraction creates additional `.npz` files

## 🔗 Related Documentation

- **Research Pipeline**: See `research_pipeline/README.md`
- **Baseline Experiments**: See `research_pipeline/baseline/README.md`
- **Quick Start**: See `research_pipeline/QUICK_START.md`

## 📚 Citation

If using this dataset for research, please cite:

> Design and Validation of HindiSER: Speech Emotion Recognition Dataset for Hindi Language

## ⚠️ Important Notes

- **Path References**: The research pipeline expects the dataset at `Dataset/my Dataset/`
- **Case Sensitivity**: Emotion folder names are case-sensitive (lowercase)
- **File Format**: Only `.wav` files are processed
- **Permissions**: Ensure read permissions for all audio files

---

**Last Updated**: Based on dataset analysis from baseline experiments
**Dataset Version**: Primary dataset (my Dataset) - 3,200 samples, 8 classes
