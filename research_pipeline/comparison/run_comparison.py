"""
Main script to run complete model comparison pipeline
Extracts features (if needed) and trains all models for comparison
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.extract_baseline_features import main as extract_features
from comparison.compare_models import compare_all_models


def main(epochs=100, batch_size=32, skip_existing=False, force=False):
    """Run complete comparison pipeline"""
    print("=" * 80)
    print("MULTI-MODEL COMPARISON PIPELINE")
    print("=" * 80)
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / 'Dataset' / 'my Dataset'
    feature_path = script_dir.parent / 'data' / 'baseline_features.npz'
    
    # Check if features exist
    if not feature_path.exists():
        print("\n[STEP 1/2] Features not found. Extracting features...")
        print("=" * 80)
        extract_features()
        print("\nFeature extraction complete!")
    else:
        print("\n[STEP 1/2] Features found. Skipping extraction...")
        print(f"Using features from: {feature_path}")
    
    # Run model comparison
    print("\n[STEP 2/2] Training and comparing all models...")
    print("=" * 80)
    
    compare_all_models(
        data_path=str(feature_path),
        results_base_dir=str(script_dir.parent / 'results'),
        epochs=epochs,
        batch_size=batch_size,
        random_seed=42,
        skip_existing=skip_existing,
        force_retrain=force
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nResults are available in:")
    print(f"  - Individual model results: research_pipeline/results/<model_name>/")
    print(f"  - Comparison results: research_pipeline/results/comparison/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete model comparison pipeline')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs per model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--skip_features', action='store_true',
                       help='Skip feature extraction if features exist')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip models that have already been trained')
    parser.add_argument('--force', action='store_true',
                       help='Force retrain even if model exists (overrides --skip_existing)')
    
    args = parser.parse_args()
    
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        force=args.force
    )
