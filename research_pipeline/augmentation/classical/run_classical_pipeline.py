"""
Classical Augmentation Pipeline
Orchestrates both generation and training stages
Can run both sequentially or run stages independently
"""

import os
import sys
from pathlib import Path

# Add paths
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent.parent
sys.path.insert(0, str(base_dir))

from augmentation.classical.generate_classical_augmented_data import generate_classical_augmented_data
from augmentation.classical.train_classical_models import train_classical_models


def run_classical_pipeline(
    stage='both',
    epochs=100,
    batch_size=32,
    skip_existing=False,
    force=False,
):
    """
    Run classical augmentation pipeline
    
    Args:
        stage: 'generation', 'training', or 'both' (default)
        epochs: Number of training epochs
        batch_size: Batch size
        skip_existing: Skip already-trained models
        force: Force retrain/regenerate
    """
    
    # From augmentation/classical/ go up 4 levels to Capstone root
    capstone_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = capstone_root / 'Dataset' / 'my Dataset'
    output_path = capstone_root / 'research_pipeline' / 'data' / 'classical_augmented_features.npz'

    if stage in ('generation', 'both'):
        print("\n" + "=" * 80)
        print("GENERATION STAGE: Classical Augmentation")
        print("=" * 80)
        
        if output_path.exists() and not force:
            print(f"\nAugmented features already exist: {output_path}")
            print("Use --force to regenerate")
        else:
            generate_classical_augmented_data(
                data_dir=data_dir,
                output_path=output_path,
                random_state=42,
            )

    if stage in ('training', 'both'):
        print("\n" + "=" * 80)
        print("TRAINING STAGE: Classical Augmentation Model Comparison")
        print("=" * 80)
        
        train_classical_models(
            epochs=epochs,
            batch_size=batch_size,
            skip_existing=skip_existing,
            force=force,
        )

    print("\n" + "=" * 80)
    print("CLASSICAL PIPELINE COMPLETE")
    print("=" * 80)


def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Classical Augmentation Pipeline (Generation + Training)'
    )
    parser.add_argument(
        '--stage',
        choices=['generation', 'training', 'both'],
        default='both',
        help='Which stage to run (default: both)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
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
    
    run_classical_pipeline(
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        force=args.force,
    )


if __name__ == '__main__':
    main()
