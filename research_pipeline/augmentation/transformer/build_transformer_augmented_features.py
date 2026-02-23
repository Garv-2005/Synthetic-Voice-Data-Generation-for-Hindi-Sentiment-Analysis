"""
Build transformer-augmented feature set: original train + MAE-generated spectrograms.
Uses same val/test as baseline. Saves .npz in baseline format.
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))
from augmentation.transformer.mae_spectrogram import (
    build_mae_model,
    patchify,
    unpatchify,
    NUM_PATCHES,
    PATCH_DIM,
    MEL_BINS,
    MAX_PAD_LEN,
)


def load_baseline_split(base_dir):
    data = np.load(base_dir / "data" / "baseline_features.npz", allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "y_test": data["y_test"],
        "emotion_map": data["emotion_map"].item(),
    }


def build_transformer_augmented_features(
    base_dir,
    output_path,
    mae_weights_path,
    num_generated_per_class=50,
    random_seed=42,
):
    """
    Load baseline split, load MAE, generate new spectrograms (with labels by round-robin),
    concatenate to train, save .npz.
    """
    base_dir = Path(base_dir)
    data = load_baseline_split(base_dir)
    X_train = data["X_train"]
    y_train = data["y_train"]
    emotion_map = data["emotion_map"]
    num_classes = len(emotion_map)
    if X_train.ndim == 4:
        X_train_2d = X_train[..., 0]
    else:
        X_train_2d = X_train
    if X_train_2d.max() > 1.0:
        scale = X_train_2d.max() - X_train_2d.min() + 1e-8
        min_val = X_train_2d.min()
    else:
        scale = 1.0
        min_val = 0.0

    # Load MAE
    model = build_mae_model()
    model(np.random.randn(1, NUM_PATCHES, PATCH_DIM).astype(np.float32))
    model.load_weights(mae_weights_path)

    total_gen = num_generated_per_class * num_classes
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    gen_patches = model.generate(total_gen, seed=random_seed)
    gen_spec = unpatchify(gen_patches.astype(np.float32))
    # Clip and scale to match baseline range
    gen_spec = np.clip(gen_spec, 0, 1)
    if scale != 1.0:
        gen_spec = gen_spec * scale + min_val
    # Labels: round-robin by class
    gen_labels = np.array([i % num_classes for i in range(total_gen)])

    # Add channel dim
    gen_spec = gen_spec[..., np.newaxis]
    X_train_new = np.concatenate([X_train, gen_spec], axis=0)
    y_train_new = np.concatenate([y_train, gen_labels], axis=0)
    shuffle_idx = np.random.permutation(len(X_train_new))
    X_train_new = X_train_new[shuffle_idx]
    y_train_new = y_train_new[shuffle_idx]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X_train=X_train_new,
        X_val=data["X_val"],
        X_test=data["X_test"],
        y_train=y_train_new,
        y_val=data["y_val"],
        y_test=data["y_test"],
        emotion_map=emotion_map,
    )
    print(f"Train: {len(X_train)} original + {total_gen} generated = {len(X_train_new)}")
    print(f"Saved: {output_path}")
    return str(output_path)


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    mae_dir = base_dir / "results" / "augmentation" / "transformer" / "mae_weights"
    mae_weights = mae_dir / "mae_weights.keras"
    if not mae_weights.exists():
        print("MAE weights not found. Run train_mae.py first.")
        return
    output_path = base_dir / "data" / "transformer_augmented_features.npz"
    build_transformer_augmented_features(
        base_dir=base_dir,
        output_path=str(output_path),
        mae_weights_path=str(mae_weights),
        num_generated_per_class=50,
    )


if __name__ == "__main__":
    main()
