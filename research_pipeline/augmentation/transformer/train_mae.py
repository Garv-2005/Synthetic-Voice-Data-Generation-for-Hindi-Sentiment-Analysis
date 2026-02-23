"""
Train MAE on baseline training spectrograms. Saves weights for generation.
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
    TIME_DIM,
)


def load_baseline_train(base_dir):
    """Load baseline .npz and return X_train (no channel), y_train, emotion_map."""
    path = base_dir / "data" / "baseline_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run baseline feature extraction first: {path}")
    data = np.load(path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    emotion_map = data["emotion_map"].item()
    # Remove channel dim for MAE: (N, 128, 174, 1) -> (N, 128, 174)
    if X_train.ndim == 4:
        X_train = X_train[..., 0]
    # Normalize to [0,1] or keep db scale
    if X_train.max() > 1.0:
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-8)
    return X_train, y_train, emotion_map


def train_mae(
    base_dir,
    save_dir,
    epochs=30,
    batch_size=32,
    random_seed=42,
):
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    X_train, y_train, _ = load_baseline_train(base_dir)
    patches = patchify(X_train.astype(np.float32))
    print(f"Patches shape: {patches.shape}")

    model = build_mae_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(save_dir / "mae_weights.keras"),
        monitor="loss",
        save_best_only=True,
    )
    model.fit(
        patches,
        patches,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[ckpt],
        verbose=1,
    )
    print(f"MAE weights saved to {save_dir}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_dir", type=str, default=None)
    args = p.parse_args()
    base_dir = Path(__file__).resolve().parent.parent.parent
    save_dir = args.save_dir or str(base_dir / "results" / "augmentation" / "transformer" / "mae_weights")
    train_mae(base_dir, save_dir, epochs=args.epochs, batch_size=args.batch_size)
