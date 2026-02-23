"""
Train VAE on baseline training spectrograms. Saves weights for generation.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))
from augmentation.vae_gan.vae_spectrogram import build_vae_model, MEL_BINS, MAX_PAD_LEN


def load_baseline_train(base_dir):
    path = base_dir / "data" / "baseline_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run baseline feature extraction first: {path}")
    data = np.load(path, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)
    if X_train.max() > 1.0:
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-8)
    return X_train


def train_vae(base_dir, save_dir, epochs=50, batch_size=32, random_seed=42):
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    X_train = load_baseline_train(base_dir)
    print(f"Train shape: {X_train.shape}")

    model = build_vae_model(input_shape=(MEL_BINS, MAX_PAD_LEN, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(save_dir / "vae_weights.keras"),
        monitor="loss",
        save_best_only=True,
    )
    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[ckpt],
        verbose=1,
    )
    print(f"VAE weights saved to {save_dir}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_dir", type=str, default=None)
    args = p.parse_args()
    save_dir = args.save_dir or str(base_dir / "results" / "augmentation" / "vae" / "vae_weights")
    train_vae(base_dir, save_dir, epochs=args.epochs, batch_size=args.batch_size)
