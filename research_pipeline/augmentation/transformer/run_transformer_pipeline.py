"""
Transformer (MAE) Augmentation Pipeline
Orchestrates both generation and training stages
Can run both sequentially or run stages independently
"""

import sys
from pathlib import Path

# Add paths
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent.parent
sys.path.insert(0, str(base_dir))

from augmentation.transformer.generate_transformer_augmented_data import generate_transformer_augmented_data
from augmentation.transformer.train_transformer_models import train_transformer_models


def run_transformer_pipeline(
    stage='both',
    mae_epochs=30,
    epochs=100,
    batch_size=32,
    num_generated_per_class=50,
    skip_existing=False,
    force=False,
):
    """
    Run Transformer augmentation pipeline
    
    Args:
        stage: 'generation', 'training', or 'both' (default)
        mae_epochs: Number of MAE training epochs
        epochs: Number of model training epochs
        batch_size: Batch size
        num_generated_per_class: Number of synthetic samples per class
        skip_existing: Skip already-trained models
        force: Force regenerate/retrain
    """

    if stage in ('generation', 'both'):
        print("\n" + "=" * 80)
        print("GENERATION STAGE: Transformer (MAE) Augmentation")
        print("=" * 80)
        
        data_dir = base_dir / "data" / "transformer_augmented_features.npz"
        if data_dir.exists() and not force:
            print(f"\nAugmented features already exist")
            print("Use --force to regenerate")
        else:
            generate_transformer_augmented_data(
                mae_epochs=mae_epochs,
                batch_size=batch_size,
                num_generated_per_class=num_generated_per_class,
                random_seed=42,
            )

    if stage in ('training', 'both'):
        print("\n" + "=" * 80)
        print("TRAINING STAGE: Transformer (MAE) Augmentation Model Comparison")
        print("=" * 80)
        
        train_transformer_models(
            epochs=epochs,
            batch_size=batch_size,
            skip_existing=skip_existing,
            force=force,
        )

    print("\n" + "=" * 80)
    print("TRANSFORMER PIPELINE COMPLETE")
    print("=" * 80)


def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Transformer (MAE) Augmentation Pipeline (Generation + Training)'
    )
    parser.add_argument(
        '--stage',
        choices=['generation', 'training', 'both'],
        default='both',
        help='Which stage to run (default: both)'
    )
    parser.add_argument(
        '--mae_epochs',
        type=int,
        default=30,
        help='Number of MAE training epochs'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of model training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num_generated_per_class',
        type=int,
        default=50,
        help='Number of synthetic samples per class'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip already-trained models'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regenerate/retrain'
    )
    
    args = parser.parse_args()
    
    run_transformer_pipeline(
        stage=args.stage,
        mae_epochs=args.mae_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_generated_per_class=args.num_generated_per_class,
        skip_existing=args.skip_existing,
        force=args.force,
    )


if __name__ == '__main__':
    main()
