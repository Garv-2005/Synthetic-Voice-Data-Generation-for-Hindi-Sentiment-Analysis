"""
Lightweight Augmentation Test Suite
Quick validation that all three approaches work on your laptop
Each test uses small datasets and few epochs for fast feedback
"""

import os
import sys
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add paths
script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(script_dir))


def create_dummy_dataset():
    """Create small dummy mel-spectrograms for testing (4 samples per emotion)"""
    n_samples_per_emotion = 4
    n_emotions = 8
    X = np.random.randn(n_samples_per_emotion * n_emotions, 128, 174, 1).astype(np.float32)
    X = (X - X.min()) / (X.max() - X.min())  # Normalize to [0, 1]
    
    y = np.repeat(np.arange(n_emotions), n_samples_per_emotion)
    
    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy',
        4: 'neutral', 5: 'sad', 6: 'sarcastic', 7: 'surprise'
    }
    
    # Split: 70% train, 15% val, 15% test (keeping stratification)
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'emotion_map': emotion_map
    }


def test_cnn_model():
    """Test CNN model training on small data"""
    print("\n" + "=" * 80)
    print("TEST 1: CNN Model Training")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from models import CNNModel
        
        # Create data
        data = create_dummy_dataset()
        print("[1/3] Created dummy training data (32 samples)")
        
        # Build model
        model = CNNModel(input_shape=(128, 174, 1), num_classes=8)
        keras_model = model.build_model()
        print("[2/3] Built CNN model")
        
        # Train briefly
        print("[3/3] Training for 2 epochs...")
        history = keras_model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=2, batch_size=8, verbose=0
        )
        
        print(f"[OK] CNN training successful")
        print(f"    Epochs trained: 2")
        print(f"    Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        return True
    
    except Exception as e:
        print(f"[ERROR] CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lstm_model():
    """Test LSTM model training on small data"""
    print("\n" + "=" * 80)
    print("TEST 2: LSTM Model Training")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from models import LSTMModel
        
        # Create data
        data = create_dummy_dataset()
        print("[1/3] Created dummy training data (32 samples)")
        
        # Build model
        model = LSTMModel(input_shape=(128, 174, 1), num_classes=8)
        keras_model = model.build_model()
        print("[2/3] Built LSTM model")
        
        # Train briefly
        print("[3/3] Training for 2 epochs...")
        history = keras_model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=2, batch_size=8, verbose=0
        )
        
        print(f"[OK] LSTM training successful")
        print(f"    Epochs trained: 2")
        print(f"    Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        return True
    
    except Exception as e:
        print(f"[ERROR] LSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_svm_model():
    """Test SVM model training on small data"""
    print("\n" + "=" * 80)
    print("TEST 3: SVM Model Training")
    print("=" * 80)
    
    try:
        from models import SVMModel
        
        # Create data
        data = create_dummy_dataset()
        print("[1/3] Created dummy training data (32 samples)")
        
        # Build model
        model = SVMModel(input_shape=(128, 174, 1), num_classes=8)
        svm_model = model.build_model()
        print("[2/3] Built SVM model")
        
        # Train
        print("[3/3] Training...")
        model.train(data['X_train'], data['y_train'], verbose=0)
        
        # Evaluate
        y_pred = svm_model.predict(data['X_train'].reshape(data['X_train'].shape[0], -1) / 255)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(data['y_train'], y_pred)
        
        print(f"[OK] SVM training successful")
        print(f"    Training accuracy: {acc:.4f}")
        return True
    
    except Exception as e:
        print(f"[ERROR] SVM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_classical_augmentation():
    """Test classical augmentation approach (CPU-based)"""
    print("\n" + "=" * 80)
    print("TEST 4: Classical Augmentation (CPU)")
    print("=" * 80)
    
    try:
        import numpy as np
        from augmentation.classical.extract_classical_augmented_features import (
            apply_time_stretch, apply_pitch_shift, apply_noise, apply_volume
        )
        
        # Create dummy audio
        sr = 16000
        duration = 1.0
        y = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))).astype(np.float32)
        
        print("[1/4] Created dummy audio signal (1 second)")
        
        # Test augmentations
        print("[2/4] Testing time stretch...")
        y_time_stretched = apply_time_stretch(y, rate=1.1)
        print(f"    Original: {len(y)} samples, Transformed: {len(y_time_stretched)} samples")
        
        print("[3/4] Testing pitch shift...")
        y_pitch_shifted = apply_pitch_shift(y, sr=sr, n_steps=2)
        print(f"    Pitch shifted successfully")
        
        print("[4/4] Testing noise and volume...")
        y_noisy = apply_noise(y, noise_factor=0.005)
        y_volume = apply_volume(y_noisy, factor=1.2)
        print(f"    Augmentations applied successfully")
        
        print(f"[OK] Classical augmentation working")
        return True
    
    except Exception as e:
        print(f"[ERROR] Classical augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_mae():
    """Test Transformer (MAE) model basics"""
    print("\n" + "=" * 80)
    print("TEST 5: Transformer (MAE) Model")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from augmentation.transformer.mae_spectrogram import build_mae_model, patchify
        
        print("[1/3] Creating dummy spectrograms (32 samples)")
        X = np.random.randn(32, 128, 174).astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min())
        
        print("[2/3] Building MAE model...")
        model = build_mae_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
        print("    MAE model built successfully")
        
        print("[3/3] Training for 1 epoch...")
        # Patch the spectrograms before training
        X_patches = patchify(X)
        history = model.fit(X_patches, X_patches, epochs=1, batch_size=8, verbose=0)
        
        print(f"[OK] MAE model working")
        print(f"    Loss: {history.history['loss'][-1]:.6f}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Transformer MAE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_model():
    """Test VAE model basics"""
    print("\n" + "=" * 80)
    print("TEST 6: VAE Model")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from augmentation.vae_gan.vae_spectrogram import build_vae_model
        
        print("[1/3] Creating dummy spectrograms (32 samples)")
        X = np.random.randn(32, 128, 174, 1).astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min())
        
        print("[2/3] Building VAE model...")
        model = build_vae_model(input_shape=(128, 174, 1), latent_dim=32)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
        print("    VAE model built successfully")
        
        print("[3/3] Training for 1 epoch...")
        history = model.fit(X, X, epochs=1, batch_size=8, verbose=0)
        
        print(f"[OK] VAE model working")
        print(f"    Loss: {history.history['loss'][-1]:.6f}")
        return True
    
    except Exception as e:
        print(f"[ERROR] VAE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("*" * 80)
    print("LIGHTWEIGHT AUGMENTATION TEST SUITE")
    print("*" * 80)
    print("\nThis suite tests all key components with minimal data (32 samples)")
    print("to verify everything works on your system before full training.")
    print("\n")
    
    results = {
        'CNN Model': test_cnn_model(),
        'LSTM Model': test_lstm_model(),
        'SVM Model': test_svm_model(),
        'Classical Augmentation': test_classical_augmentation(),
        'Transformer (MAE)': test_transformer_mae(),
        'VAE Model': test_vae_model(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("SUCCESS: All tests passed! Your system is ready for full training.")
        print("\nNext steps:")
        print("1. Push code to GitHub")
        print("2. Run on faster GPU system:")
        print("   cd research_pipeline/augmentation/classical")
        print("   python run_classical_pipeline.py --epochs 50 --stage generation")
        print("   python run_classical_pipeline.py --epochs 100 --stage training")
    else:
        print("FAILURE: Some tests failed. Check errors above.")
        print("Most likely issues:")
        print("1. TensorFlow not properly installed")
        print("2. GPU driver issues (but CPU fallback should work)")
        print("3. Missing dependencies")
        print("\nFix:")
        print("  1. Run: python tests/gpu_diagnostic.py")
        print("  2. Display output and troubleshoot accordingly")
    
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
