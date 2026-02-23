"""
Run full VAE augmentation pipeline:
  Ensure baseline features -> Train VAE (if needed) -> Build augmented features -> Run 6-model comparison.
Results: results/augmentation/vae/
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent.parent
sys.path.insert(0, str(base_dir))

from baseline.extract_baseline_features import main as baseline_extract_main
from augmentation.vae_gan.train_vae import train_vae
from augmentation.vae_gan.build_vae_augmented_features import build_vae_augmented_features
from comparison.compare_models import compare_all_models


def run_vae_pipeline(
    vae_epochs=50,
    epochs=100,
    batch_size=32,
    skip_existing=False,
    force=False,
):
    data_dir = base_dir / "data"
    baseline_npz = data_dir / "baseline_features.npz"
    vae_weights_dir = base_dir / "results" / "augmentation" / "vae" / "vae_weights"
    vae_weights = vae_weights_dir / "vae_weights.keras"
    vae_npz = data_dir / "vae_augmented_features.npz"
    results_base = base_dir / "results" / "augmentation" / "vae"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VAE AUGMENTATION PIPELINE")
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

    if not vae_weights.exists():
        print("\n[STEP 1/3] Training VAE...")
        train_vae(base_dir, str(vae_weights_dir), epochs=vae_epochs, batch_size=batch_size)
    else:
        print("\n[STEP 1/3] VAE weights found. Skipping VAE training.")

    if not vae_npz.exists() or force:
        print("\n[STEP 2/3] Building VAE-augmented features...")
        build_vae_augmented_features(
            base_dir=base_dir,
            output_path=str(vae_npz),
            vae_weights_path=str(vae_weights),
            num_generated_per_class=50,
        )
    else:
        print("\n[STEP 2/3] Using existing VAE-augmented features.")

    print("\n[STEP 3/3] Training and comparing all models...")
    compare_all_models(
        data_path=str(vae_npz),
        results_base_dir=str(results_base),
        epochs=epochs,
        batch_size=batch_size,
        random_seed=42,
        skip_existing=skip_existing,
        force_retrain=force,
    )

    print("\n" + "=" * 80)
    print("VAE PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Results: {results_base}")
    print(f"  Per-model: .../cnn/, lstm/, cnn_lstm/, resnet/, transformer/, svm/")
    print(f"  Comparison: {results_base}/comparison/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vae_epochs", type=int, default=50)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run_vae_pipeline(
        vae_epochs=args.vae_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        force=args.force,
    )
