# Approach 3: Advanced Generative Models (VAE & GAN)

Generate new spectrograms using VAE (and optionally GAN) for augmentation.

## VAE

- **Encoder:** Compresses spectrogram to latent vector.
- **Decoder:** Reconstructs spectrogram from latent.
- **Generation:** Sample new z from N(0,1), decode to new spectrograms.

## GAN (optional / future)

- **Generator:** Maps noise to spectrogram.
- **Discriminator:** Real vs fake.
- Placeholder structure is in place; full training can be added later.

## Scripts

| Script | Purpose |
|--------|--------|
| `vae_spectrogram.py` | VAE model (encoder + decoder). |
| `train_vae.py` | Train VAE on baseline train spectrograms. |
| `build_vae_augmented_features.py` | Add VAE-generated spectrograms to train, save .npz. |
| `run_vae_pipeline.py` | Train VAE -> build features -> run 6-model comparison. |

## Output layout

- **VAE weights:** `../results/augmentation/vae/vae_weights/`
- **Features:** `../data/vae_augmented_features.npz`
- **Results:** `../results/augmentation/vae/cnn/`, ... `comparison/`

## Usage

```bash
cd research_pipeline/augmentation/vae_gan
python run_vae_pipeline.py
python run_vae_pipeline.py --skip_existing --vae_epochs 50
```
