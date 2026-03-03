"""
GENERATION STAGE: Transformer (MAE) Augmentation
Trains a Masked Autoencoder (MAE) on baseline training spectrograms,
then generates synthetic spectrograms by sampling from the learned representation.
Combines original + generated training data.

Input: research_pipeline/data/baseline_features.npz (from Phase 1)
Output: research_pipeline/data/transformer_augmented_features.npz
        research_pipeline/results/augmentation/transformer/mae_weights/

GPU: Required (Transformer training is GPU-intensive)
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add paths
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))

from augmentation.transformer.train_mae import train_mae
from augmentation.transformer.build_transformer_augmented_features import build_transformer_augmented_features


def generate_transformer_augmented_data(
    mae_epochs=30,
    batch_size=32,
    num_generated_per_class=50,
    random_seed=42,
):
    """
    Generation stage for Transformer augmentation:
    1. Ensure baseline features exist (extract if needed)
    2. Train MAE on training spectrograms
    3. Generate + combine with original training data
    """
    
    base_dir_path = base_dir
    capstone_root = Path(__file__).resolve().parent.parent.parent.parent  # Go up 4 levels
    data_dir = capstone_root / 'research_pipeline' / "data"
    baseline_npz = data_dir / "baseline_features.npz"
    mae_weights_dir = base_dir_path / "results" / "augmentation" / "transformer" / "mae_weights"
    mae_weights = mae_weights_dir / "mae_weights.keras"
    transformer_npz = data_dir / "transformer_augmented_features.npz"

    data_dir.mkdir(parents=True, exist_ok=True)
    mae_weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATION STAGE: Transformer (MAE) Augmentation")
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
        print("✓ Baseline features extracted")
    else:
        print(f"✓ Baseline features found: {baseline_npz}")

    # Step 2: Train MAE (unless weights exist)
    print("\n[2/3] Training Masked Autoencoder (MAE)...")
    if mae_weights.exists():
        print(f"✓ MAE weights already exist: {mae_weights}")
        print("Use --force to retrain MAE")
    else:
        print("Training MAE on baseline training spectrograms...")
        train_mae(
            base_dir=base_dir_path,
            save_dir=str(mae_weights_dir),
            epochs=mae_epochs,
            batch_size=batch_size,
            random_seed=random_seed,
        )
        print("✓ MAE training complete")

    # Step 3: Generate synthetic spectrograms
    print("\n[3/3] Generating synthetic spectrograms...")
    if transformer_npz.exists():
        print(f"✓ Augmented features already exist: {transformer_npz}")
        print("Use --force to regenerate")
    else:
        build_transformer_augmented_features(
            base_dir=base_dir_path,
            output_path=str(transformer_npz),
            mae_weights_path=str(mae_weights),
            num_generated_per_class=num_generated_per_class,
            random_seed=random_seed,
        )
        print("✓ Synthetic spectrograms generated and combined with training data")

    print("\n" + "=" * 80)
    print("TRANSFORMER GENERATION STAGE COMPLETE")
    print("=" * 80)
    print(f"\nGenerated features: {transformer_npz}")
    print(f"MAE weights: {mae_weights}")
    print(f"\nNext: Run training stage with:")
    print(f"  python train_transformer_models.py")


def main():
    """Entry point for generation stage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Transformer (MAE) generation stage'
    )
    parser.add_argument(
        '--mae_epochs',
        type=int,
        default=30,
        help='Number of MAE training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for MAE training'
    )
    parser.add_argument(
        '--num_generated_per_class',
        type=int,
        default=50,
        help='Number of synthetic samples to generate per class'
    )
    
    args = parser.parse_args()
    
    generate_transformer_augmented_data(
        mae_epochs=args.mae_epochs,
        batch_size=args.batch_size,
        num_generated_per_class=args.num_generated_per_class,
        random_seed=42,
    )


if __name__ == '__main__':
    main()
