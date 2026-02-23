# Approach 2: Transformer-Based (MAE) Spectrogram Generation

Masked Autoencoder (MAE) on spectrograms: tokenize into patches, mask a portion, train to reconstruct. Then generate new spectrograms by iterative masking and prediction.

## Concept

1. **Tokenize:** Split spectrogram into patches (e.g. 16×18), flatten to sequence.
2. **Train:** Mask random patches, train Transformer to predict masked content.
3. **Generate:** Start from random/noisy spectrogram, repeatedly mask and predict to get new samples.

## Scripts

| Script | Purpose |
|--------|--------|
| `mae_spectrogram.py` | MAE model (patch embed, encoder, decoder). |
| `train_mae.py` | Train MAE on baseline train spectrograms. |
| `build_transformer_augmented_features.py` | Load baseline split, add MAE-generated spectrograms to train, save `.npz`. |
| `run_transformer_pipeline.py` | Train MAE (if needed) -> build augmented features -> run 6-model comparison. |

## Output layout

- **MAE weights:** `../results/augmentation/transformer/mae_weights/`
- **Features:** `../data/transformer_augmented_features.npz`
- **Results:** `../results/augmentation/transformer/cnn/`, `lstm/`, ... `comparison/`

## Usage

```bash
cd research_pipeline/augmentation/transformer
python run_transformer_pipeline.py
python run_transformer_pipeline.py --skip_existing --mae_epochs 30
```

## Dependencies

Same as baseline (TensorFlow, numpy, etc.). No extra packages.
