"""
GENERATION STAGE: VAE Augmentation
Trains a Variational Autoencoder on baseline training spectrograms,
then samples from the learned latent distribution to generate synthetic spectrograms.
Combines original + generated training data.

Input: research_pipeline/data/baseline_features.npz (from Phase 1)
Output: research_pipeline/data/vae_augmented_features.npz
        research_pipeline/results/augmentation/vae/vae_weights/

GPU: Required (VAE training is GPU-intensive)
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add paths
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))

from augmentation.vae_gan.train_vae import train_vae
from augmentation.vae_gan.build_vae_augmented_features import build_vae_augmented_features


def generate_vae_augmented_data(
    vae_epochs=50,
    batch_size=32,
    num_generated_per_class=50,
    random_seed=42,
):
    """
    Generation stage for VAE augmentation:
    1. Ensure baseline features exist (extract if needed)
    2. Train VAE on training spectrograms
    3. Sample from latent + generate + combine with original training data
    """
    
    base_dir_path = base_dir
    capstone_root = Path(__file__).resolve().parent.parent.parent.parent  # Go up 4 levels
    data_dir = capstone_root / 'research_pipeline' / "data"
    baseline_npz = data_dir / "baseline_features.npz"
    vae_weights_dir = base_dir_path / "results" / "augmentation" / "vae" / "vae_weights"
    vae_weights = vae_weights_dir / "vae_weights.keras"
    vae_npz = data_dir / "vae_augmented_features.npz"

    data_dir.mkdir(parents=True, exist_ok=True)
    vae_weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATION STAGE: VAE Augmentation")
    print("=" * 80)

    # Step 1: Ensure baseline features exist
    print("\n[1/3] Checking baseline features...")
    if not baseline_npz.exists():
        print("Baseline features not found. Extracting baseline features...")
        # Lazy import to avoid matplotlib hang
        from baseline.extract_baseline_features import main as baseline_extract_main
        orig = os.getcwd()
        try:
            os.chdir(base_dir_path / "baseline")
            baseline_extract_main()
        finally:
            os.chdir(orig)
        print("[OK] Baseline features extracted")
    else:
        print(f"[OK] Baseline features found: {baseline_npz}")

    # Step 2: Train VAE (unless weights exist)
    print("\n[2/3] Training Variational Autoencoder (VAE)...")
    if vae_weights.exists():
        print(f"[OK] VAE weights already exist: {vae_weights}")
        print("Use --force to retrain VAE")
    else:
        print("Training VAE on baseline training spectrograms...")
        train_vae(
            base_dir=base_dir_path,
            save_dir=str(vae_weights_dir),
            epochs=vae_epochs,
            batch_size=batch_size,
            random_seed=random_seed,
        )
        print("[OK] VAE training complete")

    # Step 3: Generate synthetic spectrograms
    print("\n[3/3] Generating synthetic spectrograms...")
    if vae_npz.exists():
        print(f"[OK] Augmented features already exist: {vae_npz}")
        print("Use --force to regenerate")
    else:
        build_vae_augmented_features(
            base_dir=base_dir_path,
            output_path=str(vae_npz),
            vae_weights_path=str(vae_weights),
            num_generated_per_class=num_generated_per_class,
            random_seed=random_seed,
        )
        print("[OK] Synthetic spectrograms generated and combined with training data")

    print("\n" + "=" * 80)
    print("VAE GENERATION STAGE COMPLETE")
    print("=" * 80)
    print(f"\nGenerated features: {vae_npz}")
    print(f"VAE weights: {vae_weights}")
    print(f"\nNext: Run training stage with:")
    print(f"  python train_vae_models.py")


def main():
    """Entry point for generation stage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='VAE generation stage'
    )
    parser.add_argument(
        '--vae_epochs',
        type=int,
        default=50,
        help='Number of VAE training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for VAE training'
    )
    parser.add_argument(
        '--num_generated_per_class',
        type=int,
        default=50,
        help='Number of synthetic samples to generate per class'
    )
    
    args = parser.parse_args()
    
    generate_vae_augmented_data(
        vae_epochs=args.vae_epochs,
        batch_size=args.batch_size,
        num_generated_per_class=args.num_generated_per_class,
        random_seed=42,
    )


if __name__ == '__main__':
    main()
