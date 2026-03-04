import numpy as np
import tensorflow as tf
from augmentation.transformer.mae_spectrogram import build_mae_model, patchify

print('Testing MAE fix...')
X = np.random.randn(8, 128, 174).astype(np.float32)
X = (X - X.min()) / (X.max() - X.min())
X_patches = patchify(X)
print(f'Patches shape: {X_patches.shape}')
model = build_mae_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
history = model.fit(X_patches, X_patches, epochs=1, batch_size=8, verbose=0)
print(f'SUCCESS: MAE training completed, loss: {history.history["loss"][-1]:.6f}')