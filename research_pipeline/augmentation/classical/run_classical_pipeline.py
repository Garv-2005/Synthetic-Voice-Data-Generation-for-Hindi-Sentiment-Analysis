"""
Run full classical-augmentation pipeline: extract features (if needed) -> train all 6 models -> comparison.
Results go to results/augmentation/classical/ for clear segregation.
"""

import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent.parent  # research_pipeline
sys.path.insert(0, str(base_dir))

from augmentation.classical.extract_classical_augmented_features import extract_classical_augmented_features, main as extract_main
from comparison.compare_models import compare_all_models


def run_classical_pipeline(
    epochs=100,
    batch_size=32,
    skip_existing=False,
    force=False,
):
    data_dir = base_dir / "data"
    feature_path = data_dir / "classical_augmented_features.npz"
    results_base = base_dir / "results" / "augmentation" / "classical"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CLASSICAL AUGMENTATION PIPELINE")
    print("=" * 80)

    if not feature_path.exists():
        print("\n[STEP 1/2] Extracting classical-augmented features...")
        extract_main()
    else:
        print("\n[STEP 1/2] Using existing classical-augmented features:")
        print(f"  {feature_path}")

    print("\n[STEP 2/2] Training and comparing all models (same as baseline)...")
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
    print("CLASSICAL PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Results: {results_base}")
    print(f"  Per-model: {results_base}/cnn/, lstm/, cnn_lstm/, resnet/, transformer/, svm/")
    print(f"  Comparison: {results_base}/comparison/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run classical augmentation pipeline")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--skip_existing", action="store_true", help="Skip already-trained models")
    p.add_argument("--force", action="store_true", help="Force retrain all models")
    args = p.parse_args()
    run_classical_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        force=args.force,
    )
