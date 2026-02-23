"""
Run full transformer (MAE) augmentation pipeline:
  Ensure baseline features -> Train MAE (if needed) -> Build augmented features -> Run 6-model comparison.
Results: results/augmentation/transformer/
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent.parent
sys.path.insert(0, str(base_dir))

from baseline.extract_baseline_features import main as baseline_extract_main
from augmentation.transformer.train_mae import train_mae
from augmentation.transformer.build_transformer_augmented_features import build_transformer_augmented_features
from comparison.compare_models import compare_all_models


def run_transformer_pipeline(
    mae_epochs=30,
    epochs=100,
    batch_size=32,
    skip_existing=False,
    force=False,
):
    data_dir = base_dir / "data"
    baseline_npz = data_dir / "baseline_features.npz"
    mae_weights_dir = base_dir / "results" / "augmentation" / "transformer" / "mae_weights"
    mae_weights = mae_weights_dir / "mae_weights.keras"
    transformer_npz = data_dir / "transformer_augmented_features.npz"
    results_base = base_dir / "results" / "augmentation" / "transformer"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRANSFORMER (MAE) AUGMENTATION PIPELINE")
    print("=" * 80)

    if not baseline_npz.exists():
        print("\n[STEP 0] Baseline features not found. Extracting baseline...")
        import os
        orig = os.getcwd()
        try:
            os.chdir(base_dir / "baseline")
            baseline_extract_main()
        finally:
            os.chdir(orig)
        print("Baseline features ready.")
    else:
        print("\n[STEP 0] Baseline features found.")

    if not mae_weights.exists():
        print("\n[STEP 1/3] Training MAE...")
        train_mae(base_dir, str(mae_weights_dir), epochs=mae_epochs, batch_size=batch_size)
    else:
        print("\n[STEP 1/3] MAE weights found. Skipping MAE training.")

    if not transformer_npz.exists() or force:
        print("\n[STEP 2/3] Building transformer-augmented features...")
        build_transformer_augmented_features(
            base_dir=base_dir,
            output_path=str(transformer_npz),
            mae_weights_path=str(mae_weights),
            num_generated_per_class=50,
        )
    else:
        print("\n[STEP 2/3] Using existing transformer-augmented features.")

    print("\n[STEP 3/3] Training and comparing all models...")
    compare_all_models(
        data_path=str(transformer_npz),
        results_base_dir=str(results_base),
        epochs=epochs,
        batch_size=batch_size,
        random_seed=42,
        skip_existing=skip_existing,
        force_retrain=force,
    )

    print("\n" + "=" * 80)
    print("TRANSFORMER PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Results: {results_base}")
    print(f"  Per-model: .../cnn/, lstm/, cnn_lstm/, resnet/, transformer/, svm/")
    print(f"  Comparison: {results_base}/comparison/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mae_epochs", type=int, default=30)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run_transformer_pipeline(
        mae_epochs=args.mae_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        force=args.force,
    )
