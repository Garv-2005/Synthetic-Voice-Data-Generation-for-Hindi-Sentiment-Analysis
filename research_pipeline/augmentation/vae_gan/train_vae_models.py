"""
TRAINING STAGE: VAE Augmentation Models
Trains all 6 models (CNN, LSTM, CNN-LSTM, ResNet, Transformer, SVM) on VAE-augmented features.
Generates model comparison visualizations and results.

Input: research_pipeline/data/vae_augmented_features.npz
Output: research_pipeline/results/augmentation/vae/

GPU: Used for all model training (same as Phase 1)
"""

import os
import sys
from pathlib import Path

# Add paths
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent.parent
sys.path.insert(0, str(base_dir))

from comparison.compare_models import compare_all_models


def train_vae_models(
    epochs=100,
    batch_size=32,
    skip_existing=False,
    force=False,
):
    """Train all models on VAE augmented features"""
    
    data_dir = base_dir / "data"
    feature_path = data_dir / "vae_augmented_features.npz"
    results_base = base_dir / "results" / "augmentation" / "vae"

    print("\n" + "=" * 80)
    print("TRAINING STAGE: VAE Augmentation Model Comparison")
    print("=" * 80)

    # Check if augmented features exist
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Augmented features not found: {feature_path}\n"
            f"Please run generation stage first:\n"
            f"  python generate_vae_augmented_data.py"
        )

    print(f"\nUsing augmented features: {feature_path}")
    print(f"Results will be saved to: {results_base}/\n")

    # Train all models (using existing comparison framework)
    compare_all_models(
        data_path=str(feature_path),
        results_base_dir=str(results_base),
        epochs=epochs,
        batch_size=batch_size,
        random_seed=42,
        skip_existing=skip_existing,
        force_retrain=force,
    )

    print("\n" + "=" * 80)
    print("VAE TRAINING STAGE COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  Per-model: {results_base}/cnn/, lstm/, cnn_lstm/, resnet/, transformer/, svm/")
    print(f"  Comparison: {results_base}/comparison/")
    
    # Show where to find comparison results
    comparison_json = results_base / 'comparison' / 'comparison_results.json'
    if comparison_json.exists():
        print(f"\nComparison results: {comparison_json}")


def main():
    """Entry point for training stage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train models on VAE augmented features',
    )
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs per model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip models that have already been trained')
    parser.add_argument('--force', action='store_true',
                       help='Force retrain all models')
    
    args = parser.parse_args()
    
    train_vae_models(
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        force=args.force,
    )


if __name__ == '__main__':
    main()
