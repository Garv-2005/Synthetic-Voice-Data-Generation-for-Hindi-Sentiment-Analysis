"""
Build VAE-augmented feature set: original train + VAE-generated spectrograms.
Same val/test as baseline. Saves .npz in baseline format.
"""

from pathlib import Path
import numpy as np
import sys

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))
from augmentation.vae_gan.vae_spectrogram import build_vae_model, MEL_BINS, MAX_PAD_LEN


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


def build_vae_augmented_features(
    base_dir,
    output_path,
    vae_weights_path,
    num_generated_per_class=50,
    random_seed=42,
):
    base_dir = Path(base_dir)
    data = load_baseline_split(base_dir)
    X_train = data["X_train"]
    y_train = data["y_train"]
    emotion_map = data["emotion_map"]
    num_classes = len(emotion_map)
    total_gen = num_generated_per_class * num_classes

    model = build_vae_model(input_shape=(MEL_BINS, MAX_PAD_LEN, 1))
    model(np.random.randn(1, MEL_BINS, MAX_PAD_LEN, 1).astype(np.float32))
    model.load_weights(vae_weights_path)

    gen = model.generate(total_gen, seed=random_seed)
    if gen.max() > 1.0 or gen.min() < 0:
        gen = np.clip(gen, 0, 1)
    # Scale to match baseline if needed
    if X_train.max() > 1.0:
        scale = X_train.max() - X_train.min()
        min_val = X_train.min()
        gen = gen * scale + min_val
    gen_labels = np.array([i % num_classes for i in range(total_gen)])

    X_train_new = np.concatenate([X_train, gen], axis=0)
    y_train_new = np.concatenate([y_train, gen_labels], axis=0)
    rng = np.random.default_rng(random_seed)
    shuffle_idx = rng.permutation(len(X_train_new))
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
    vae_dir = base_dir / "results" / "augmentation" / "vae" / "vae_weights"
    vae_weights = vae_dir / "vae_weights.keras"
    if not vae_weights.exists():
        print("VAE weights not found. Run train_vae.py first.")
        return
    output_path = base_dir / "data" / "vae_augmented_features.npz"
    build_vae_augmented_features(
        base_dir=base_dir,
        output_path=str(output_path),
        vae_weights_path=str(vae_weights),
        num_generated_per_class=50,
    )


if __name__ == "__main__":
    main()
