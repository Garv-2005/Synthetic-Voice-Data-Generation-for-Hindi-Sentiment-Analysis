"""
Baseline Feature Extraction for Hindi Speech Emotion Recognition
Extracts Mel-spectrogram features from original dataset without augmentation
Includes dataset analysis and visualization
"""

import os
import sys
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from tqdm import tqdm

# Add utils to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
from utils.visualization import plot_dataset_distribution, plot_sample_spectrograms


class BaselineFeatureExtractor:
    """Extract baseline features from audio dataset"""
    
    def __init__(self, data_dir, sample_rate=16000, n_mels=128, max_pad_len=174):
        """
        Args:
            data_dir: Path to dataset directory
            sample_rate: Target sampling rate
            n_mels: Number of mel filter banks
            max_pad_len: Maximum spectrogram length (time frames)
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_pad_len = max_pad_len
        
    def extract_melspectrogram(self, audio_path):
        """Extract Mel-spectrogram from audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            y, _ = librosa.effects.trim(y)
            
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=self.n_mels,
                hop_length=512,
                n_fft=2048
            )
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Pad or truncate to fixed length
            if log_mel_spectrogram.shape[1] > self.max_pad_len:
                log_mel_spectrogram = log_mel_spectrogram[:, :self.max_pad_len]
            else:
                pad_width = self.max_pad_len - log_mel_spectrogram.shape[1]
                log_mel_spectrogram = np.pad(
                    log_mel_spectrogram, 
                    pad_width=((0, 0), (0, pad_width)), 
                    mode='constant',
                    constant_values=log_mel_spectrogram.min()
                )
            
            return log_mel_spectrogram
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_all_features(self):
        """Extract features from all audio files in dataset"""
        features = []
        labels = []
        file_paths = []
        emotion_map = {}
        label_count = 0
        
        print(f"Scanning dataset in: {self.data_dir}")
        print("=" * 60)
        
        # First pass: collect all file paths
        all_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    emotion = os.path.basename(root)
                    
                    if emotion not in emotion_map:
                        emotion_map[emotion] = label_count
                        label_count += 1
                    
                    label = emotion_map[emotion]
                    all_files.append((audio_path, emotion, label))
        
        print(f"Found {len(all_files)} audio files")
        print(f"Processing files...")
        
        # Extract features with progress bar
        with tqdm(total=len(all_files), desc="Extracting features", ncols=100,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for audio_path, emotion, label in all_files:
                mel_spec = self.extract_melspectrogram(audio_path)
                if mel_spec is not None:
                    features.append(mel_spec)
                    labels.append(label)
                    file_paths.append(audio_path)
                pbar.update(1)
                pbar.set_postfix_str(f"Processed: {len(features)}")
        
        print(f"\nTotal files processed: {len(features)}")
        print(f"Emotion classes: {len(emotion_map)}")
        print(f"Feature shape: {features[0].shape if features else 'None'}")
        
        return np.array(features), np.array(labels), emotion_map, file_paths
    
    def analyze_dataset(self, labels, emotion_map, file_paths):
        """Analyze dataset distribution and statistics"""
        class_distribution = Counter(labels)
        
        # Convert numpy int64 keys to regular Python ints for JSON serialization
        class_distribution_dict = {int(k): int(v) for k, v in class_distribution.items()}
        
        analysis = {
            'total_samples': int(len(labels)),
            'num_classes': int(len(emotion_map)),
            'emotion_distribution': {},
            'class_distribution': class_distribution_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        # Count samples per emotion
        emotion_names = {v: k for k, v in emotion_map.items()}
        for label_id, count in class_distribution.items():
            emotion_name = emotion_names[int(label_id)]  # Ensure int conversion
            analysis['emotion_distribution'][emotion_name] = int(count)
        
        return analysis
    
    def create_train_test_split(self, features, labels, test_size=0.2, 
                                validation_size=0.15, random_state=42):
        """Create train/validation/test splits"""
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_trainval
        )
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(features)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(features)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(features)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_features(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                     emotion_map, output_dir='../data'):
        """Save extracted features and labels"""
        output_dir = Path(output_dir)
        os.makedirs(str(output_dir), exist_ok=True)
        
        # Add channel dimension for CNN
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        output_path = os.path.join(output_dir, 'baseline_features.npz')
        np.savez(
            output_path,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            emotion_map=emotion_map
        )
        
        print(f"\nFeatures saved to: {output_path}")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  X_test shape: {X_test.shape}")
        
        return output_path


def main():
    """Main function for baseline feature extraction"""
    print("=" * 60)
    print("BASELINE FEATURE EXTRACTION")
    print("=" * 60)
    
    # Configuration
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up to Capstone root
    data_dir = project_root / 'Dataset' / 'my Dataset'
    output_dir = script_dir.parent / 'data'
    results_dir = script_dir.parent / 'results' / 'baseline'
    
    os.makedirs(str(output_dir), exist_ok=True)
    os.makedirs(str(results_dir), exist_ok=True)
    
    # Initialize extractor
    extractor = BaselineFeatureExtractor(
        data_dir=str(data_dir),
        sample_rate=16000,
        n_mels=128,
        max_pad_len=174
    )
    
    # Extract features
    print("\n[1/4] Extracting features...")
    features, labels, emotion_map, file_paths = extractor.extract_all_features()
    
    # Analyze dataset
    print("\n[2/4] Analyzing dataset...")
    analysis = extractor.analyze_dataset(labels, emotion_map, file_paths)
    
    # Save analysis
    analysis_path = os.path.join(results_dir, 'dataset_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Dataset analysis saved to: {analysis_path}")
    
    # Visualize dataset distribution
    print("\n[3/4] Generating visualizations...")
    plot_dataset_distribution(
        labels, 
        emotion_map, 
        save_path=os.path.join(results_dir, 'dataset_distribution.png')
    )
    
    # Plot sample spectrograms
    if len(features) > 0:
        sample_indices = [np.where(labels == i)[0][0] for i in range(len(emotion_map))]
        sample_features = features[sample_indices]
        sample_labels = labels[sample_indices]
        
        plot_sample_spectrograms(
            sample_features,
            sample_labels,
            emotion_map,
            save_path=os.path.join(results_dir, 'sample_spectrograms.png')
        )
    
    # Create splits
    print("\n[4/4] Creating train/validation/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = extractor.create_train_test_split(
        features, labels, test_size=0.2, validation_size=0.15
    )
    
    # Save features
    feature_path = extractor.save_features(
        X_train, X_val, X_test, y_train, y_val, y_test,
        emotion_map, output_dir=str(output_dir)
    )
    
    print("\n" + "=" * 60)
    print("BASELINE FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Results saved in: {results_dir}")
    print(f"Features saved in: {feature_path}")
    
    return feature_path, analysis


if __name__ == '__main__':
    feature_path, analysis = main()
