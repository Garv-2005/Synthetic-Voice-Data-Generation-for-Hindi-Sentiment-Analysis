"""
Classical Augmentation: Extract Mel-spectrograms from original + augmented audio.
Uses same split and format as baseline for fair comparison.
Val/test are unchanged; only training set is augmented (original + augmented copies).
"""

import os
import sys
from pathlib import Path
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add paths
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))
from baseline.extract_baseline_features import BaselineFeatureExtractor


def _mel_from_audio(y, sr, extractor):
    """Compute Mel-spectrogram from audio array (same params as baseline)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=extractor.n_mels,
        hop_length=512, n_fft=2048
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    max_len = extractor.max_pad_len
    if log_mel.shape[1] > max_len:
        log_mel = log_mel[:, :max_len]
    else:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(
            log_mel,
            pad_width=((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=log_mel.min()
        )
    return log_mel


def apply_time_stretch(y, rate):
    """Time stretch (rate < 1 slower, > 1 faster)."""
    return librosa.effects.time_stretch(y, rate=rate)


def apply_pitch_shift(y, sr, n_steps):
    """Pitch shift by n_steps semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def apply_noise(y, noise_factor=0.005, rng=None):
    """Add Gaussian noise."""
    rng = rng or np.random
    noise = rng.randn(len(y)).astype(np.float32) * noise_factor
    return y + noise


def apply_volume(y, factor, rng=None):
    """Scale amplitude."""
    rng = rng or np.random
    if isinstance(factor, (list, tuple)):
        low, high = factor
        factor = rng.uniform(low, high)
    return y * factor


def extract_classical_augmented_features(
    data_dir,
    output_path,
    sample_rate=16000,
    n_mels=128,
    max_pad_len=174,
    random_state=42,
    augment_per_sample=4,
    time_stretch_rates=(0.9, 1.1),
    pitch_steps=(-2, 2),
    noise_factor=0.005,
    volume_range=(0.8, 1.2),
):
    """
    Build augmented training set: original + classical augmentations.
    Val/test identical to baseline (same split).
    """
    rng = np.random.default_rng(random_state)
    extractor = BaselineFeatureExtractor(
        data_dir=str(data_dir),
        sample_rate=sample_rate,
        n_mels=n_mels,
        max_pad_len=max_pad_len,
    )

    # 1) Extract all original features and get file paths (same order as baseline)
    print("Extracting original features (same as baseline)...")
    features, labels, emotion_map, file_paths = extractor.extract_all_features()
    features = np.array(features)
    labels = np.array(labels)

    # 2) Same split as baseline
    test_size = 0.2
    validation_size = 0.15
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    val_ratio = validation_size / (1 - test_size)
    X_train_orig, X_val, y_train_orig, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state, stratify=y_trainval
    )

    # Train indices in full list (to know which files to augment)
    n = len(features)
    indices = np.arange(n)
    _, test_idx, _, _ = train_test_split(
        indices, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    trainval_idx = np.setdiff1d(indices, test_idx)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        labels[trainval_idx],
        test_size=val_ratio,
        random_state=random_state,
        stratify=labels[trainval_idx],
    )
    train_idx = np.sort(train_idx)  # for deterministic iteration

    # 3) For each training file, add augmented versions
    augmented_features = []
    augmented_labels = []
    n_aug = 0
    with tqdm(total=len(train_idx), desc="Classical augmentation", ncols=100) as pbar:
        for idx in train_idx:
            path = file_paths[idx]
            label = labels[idx]
            try:
                y, sr = librosa.load(path, sr=extractor.sample_rate)
                y, _ = librosa.effects.trim(y)
            except Exception as e:
                continue
            count = 0
            # Time stretch
            for rate in time_stretch_rates:
                y_a = apply_time_stretch(y, rate)
                mel = _mel_from_audio(y_a, sr, extractor)
                augmented_features.append(mel)
                augmented_labels.append(label)
                count += 1
            # Pitch shift
            for steps in pitch_steps:
                y_a = apply_pitch_shift(y, sr, steps)
                mel = _mel_from_audio(y_a, sr, extractor)
                augmented_features.append(mel)
                augmented_labels.append(label)
                count += 1
            # Noise
            y_a = apply_noise(y, noise_factor=noise_factor, rng=rng)
            mel = _mel_from_audio(y_a, sr, extractor)
            augmented_features.append(mel)
            augmented_labels.append(label)
            count += 1
            # Volume
            y_a = apply_volume(y, volume_range, rng=rng)
            mel = _mel_from_audio(y_a, sr, extractor)
            augmented_features.append(mel)
            augmented_labels.append(label)
            count += 1
            n_aug += count
            pbar.update(1)
    augmented_features = np.array(augmented_features)
    augmented_labels = np.array(augmented_labels)

    # 4) Combine original train + augmented
    X_train = np.concatenate([X_train_orig, augmented_features], axis=0)
    y_train = np.concatenate([y_train_orig, augmented_labels], axis=0)
    # Shuffle combined train
    shuffle_idx = rng.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    print(f"\nTrain: original {len(X_train_orig)} + augmented {len(augmented_features)} = {len(X_train)}")
    print(f"Validation: {len(X_val)} (unchanged)")
    print(f"Test: {len(X_test)} (unchanged)")

    # 5) Add channel dimension and save (same format as baseline)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        emotion_map=emotion_map,
    )
    print(f"Saved: {output_path}")
    return output_path


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    data_dir = project_root / "Dataset" / "my Dataset"
    output_path = script_dir.parent.parent / "data" / "classical_augmented_features.npz"

    print("=" * 60)
    print("CLASSICAL AUGMENTATION - FEATURE EXTRACTION")
    print("=" * 60)
    extract_classical_augmented_features(
        data_dir=str(data_dir),
        output_path=str(output_path),
        random_state=42,
    )
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    return output_path


if __name__ == "__main__":
    main()
