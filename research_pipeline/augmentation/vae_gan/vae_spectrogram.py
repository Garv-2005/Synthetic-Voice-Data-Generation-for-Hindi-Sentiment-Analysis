"""
Variational Autoencoder for spectrograms. Encoder -> latent z -> Decoder -> reconstruction.
Generation: sample z ~ N(0,1), decode to new spectrogram.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MEL_BINS = 128
MAX_PAD_LEN = 174
LATENT_DIM = 256


def build_vae_encoder(input_shape=(MEL_BINS, MAX_PAD_LEN, 1), latent_dim=LATENT_DIM):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    return keras.Model(inp, [z_mean, z_log_var], name="encoder")


def build_vae_decoder(latent_dim=LATENT_DIM, output_shape=(MEL_BINS, MAX_PAD_LEN, 1)):
    # Encoder with stride 2 three times: 128->64->32->16, 174->87->44->22
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 22 * 128, activation="relu")(inp)
    x = layers.Reshape((16, 22, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Lambda(lambda t: t[:, :MEL_BINS, :MAX_PAD_LEN, :])(x)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inp, out, name="decoder")


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()

    def call(self, x, training=False):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling((z_mean, z_log_var))
        recon = self.decoder(z)
        return recon

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling((z_mean, z_log_var))
            recon = self.decoder(z)
            recon_loss = tf.reduce_mean(keras.losses.mse(data, recon))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + 0.001 * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    @property
    def metrics(self):
        return []

    def generate(self, batch_size, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        z = tf.random.normal((batch_size, self.encoder.output[0].shape[-1]))
        return self.decoder(z).numpy()


def build_vae_model(input_shape=(MEL_BINS, MAX_PAD_LEN, 1), latent_dim=LATENT_DIM):
    encoder = build_vae_encoder(input_shape, latent_dim)
    decoder = build_vae_decoder(latent_dim, input_shape)
    vae = VAE(encoder, decoder)
    vae.build((None,) + input_shape)
    return vae
