# Approach 1: Classical Augmentation

Traditional, low-cost transformations applied to **audio** before feature extraction. This gives a stronger baseline; any generative approach should ideally outperform this.

## Techniques

- **Time stretching:** Slow down or speed up audio (no pitch change).
- **Pitch shifting:** Raise or lower pitch (no speed change).
- **Background noise:** Add noise (Gaussian or optional UrbanSound8k) for robustness.
- **Volume perturbation:** Random amplitude scaling.

## Scripts

| Script | Purpose |
|--------|--------|
| `extract_classical_augmented_features.py` | Build augmented training set; val/test unchanged. Saves `../data/classical_augmented_features.npz`. |
| `run_classical_pipeline.py` | Extract (if needed) → run same 6-model comparison → save under `../results/augmentation/classical/`. |

## Output layout

- **Features:** `research_pipeline/data/classical_augmented_features.npz` (same keys as baseline).
- **Results:** `research_pipeline/results/augmentation/classical/`
  - `cnn/`, `lstm/`, `cnn_lstm/`, `resnet/`, `transformer/`, `svm/` — per-model metrics and plots.
  - `comparison/` — comparison charts and `comparison_results.json`.

## Usage

```bash
cd research_pipeline/augmentation/classical
python run_classical_pipeline.py
python run_classical_pipeline.py --skip_existing --epochs 100
```

## Reproducibility

Uses the same random seed (42) and train/val/test split logic as the baseline so the test set is identical for fair comparison.
