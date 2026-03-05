"""
Masked Autoencoder (MAE) for spectrograms.
Patches spectrogram, optionally masks, encoder-decoder reconstructs.
Generation: sample latent and decode to new spectrograms.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Match baseline
MEL_BINS = 128
MAX_PAD_LEN = 174
PATCH_H = 16
PATCH_W = 18
TIME_PAD = (PATCH_W - (MAX_PAD_LEN % PATCH_W)) % PATCH_W
TIME_DIM = MAX_PAD_LEN + TIME_PAD
NUM_PATCHES = (MEL_BINS // PATCH_H) * (TIME_DIM // PATCH_W)
PATCH_DIM = PATCH_H * PATCH_W
LATENT_DIM = 256
NUM_HEADS = 4
FF_DIM = 512


def patchify(spec, patch_h=PATCH_H, patch_w=PATCH_W, time_dim=TIME_DIM):
    """spec: (batch, mel, time). Return (batch, num_patches, patch_dim)."""
    if spec.ndim == 2:
        spec = spec[np.newaxis, ...]
    batch = spec.shape[0]
    if spec.shape[2] < time_dim:
        pad = np.zeros((batch, spec.shape[1], time_dim - spec.shape[2]), dtype=spec.dtype)
        spec = np.concatenate([spec, pad], axis=2)
    elif spec.shape[2] > time_dim:
        spec = spec[:, :, :time_dim]
    patches = []
    for i in range(0, MEL_BINS, patch_h):
        for j in range(0, time_dim, patch_w):
            p = spec[:, i : i + patch_h, j : j + patch_w]
            patches.append(p.reshape(batch, -1))
    return np.concatenate(patches, axis=1).reshape(batch, -1, PATCH_DIM)


def unpatchify(patches, patch_h=PATCH_H, patch_w=PATCH_W, time_dim=TIME_DIM):
    """patches: (batch, num_patches, patch_dim). Return (batch, mel, time)."""
    batch = patches.shape[0]
    nph = MEL_BINS // patch_h
    npw = time_dim // patch_w
    out = np.zeros((batch, MEL_BINS, time_dim), dtype=patches.dtype)
    idx = 0
    for i in range(nph):
        for j in range(npw):
            p = patches[:, idx].reshape(batch, patch_h, patch_w)
            out[:, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w] = p
            idx += 1
    return out[:, :, :MAX_PAD_LEN]


class MAESpectrogram(keras.Model):
    """Patch-based Transformer autoencoder. Reconstructs spectrogram from patches.
    Optional: mask ratio in loss for MAE-style training."""

    def __init__(
        self,
        num_patches=NUM_PATCHES,
        patch_dim=PATCH_DIM,
        embed_dim=LATENT_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        mask_ratio=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim

        self.patch_proj = layers.Dense(embed_dim)
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, num_patches, embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.enc_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.enc_ff = layers.Dense(ff_dim, activation="gelu")
        self.enc_out = layers.Dense(embed_dim)
        self.enc_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dec_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.dec_ff = layers.Dense(ff_dim, activation="gelu")
        self.dec_out = layers.Dense(embed_dim)
        self.dec_head = layers.Dense(patch_dim)
        self.dec_norm = layers.LayerNormalization(epsilon=1e-6)

    def encode(self, patches):
        x = self.patch_proj(patches) + self.pos_embed
        attn = self.enc_layer(x, x)
        x = self.enc_norm(x + attn)
        x = x + self.enc_out(self.enc_ff(x))
        return x

    def decode(self, z):
        attn = self.dec_layer(z, z)
        x = self.dec_norm(z + attn)
        x = x + self.dec_out(self.dec_ff(x))
        return self.dec_head(x)

    def call(self, patches, training=False):
        z = self.encode(patches)
        recon = self.decode(z)
        return recon

    def generate(self, batch_size, seed=None):
        """Generate new spectrograms: sample latent from N(0,1), decode."""
        if seed is not None:
            tf.random.set_seed(seed)
        z = tf.random.normal((batch_size, self.num_patches, self.embed_dim))
        patches = self.decode(z)
        return patches.numpy()


def build_mae_model():
    return MAESpectrogram(mask_ratio=0.75)
