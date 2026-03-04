"""
GPU Diagnostic Tool
Checks TensorFlow/CUDA setup and GPU availability
Run this to verify GPU is properly configured before training
"""

import os
import sys
import platform
from pathlib import Path

# Add paths
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
sys.path.insert(0, str(base_dir))


def check_system_info():
    """Print system information"""
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()


def check_tensorflow():
    """Check TensorFlow installation and GPU support"""
    print("=" * 80)
    print("TENSORFLOW CONFIGURATION")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow version: {tf.__version__}")
        print(f"[OK] TensorFlow location: {tf.__file__}")
        
        # Check CUDA/cuDNN
        print("\nCUDA/cuDNN Configuration:")
        try:
            cuda_version = tf.sysconfig.get_build_info()['cuda_version']
            print(f"  CUDA version (used in build): {cuda_version}")
        except:
            print("  CUDA version: Not available")
        
        try:
            cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
            print(f"  cuDNN version (used in build): {cudnn_version}")
        except:
            print("  cuDNN version: Not available")
        
        # Check GPU devices
        print("\nGPU Devices:")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"    [{i}] {gpu.name}")
            print("\nGPU Details:")
            for gpu in gpus:
                print(f"  {gpu}")
        else:
            print("[CPU] No GPU detected")
        
        # Check CPU
        cpus = tf.config.list_physical_devices('CPU')
        print(f"\nCPU Devices: {len(cpus)} CPU(s)")
        for i, cpu in enumerate(cpus):
            print(f"    [{i}] {cpu.name}")
        
        print("\nBuild with CUDA support:", tf.test.is_built_with_cuda())
        
        return True, gpus
    
    except ImportError as e:
        print(f"[ERROR] TensorFlow not installed: {e}")
        return False, []


def test_gpu_training():
    """Test actual GPU training with a simple model"""
    print("\n" + "=" * 80)
    print("GPU TRAINING TEST")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        import numpy as np
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("[INFO] No GPU available, test will run on CPU")
        
        # Create simple data
        X_test = np.random.randn(100, 128, 174, 1).astype(np.float32)
        y_test = np.random.randint(0, 8, 100)
        
        print("[1/3] Building simple CNN model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 174, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("[OK] Model built")
        
        print("[2/3] Training for 2 epochs...")
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            history = model.fit(X_test, y_test, epochs=2, batch_size=32, verbose=0)
        
        print("[OK] Training completed")
        print(f"    Loss: {history.history['loss'][-1]:.4f}")
        print(f"    Accuracy: {history.history['accuracy'][-1]:.4f}")
        
        print("[3/3] Testing prediction...")
        predictions = model.predict(X_test[:5], verbose=0)
        print(f"[OK] Predictions made successfully")
        print(f"    Sample prediction shape: {predictions.shape}")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_environment_variables():
    """Check relevant environment variables"""
    print("\n" + "=" * 80)
    print("ENVIRONMENT VARIABLES")
    print("=" * 80)
    
    env_vars = [
        'CUDA_HOME',
        'CUDA_PATH',
        'CUDNN_HOME',
        'LD_LIBRARY_PATH',
        'PATH',
        'TF_CPP_MIN_LOG_LEVEL',
        'TF_FORCE_GPU_ALLOW_GROWTH',
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            if 'PATH' in var.upper():
                print(f"{var}: [set, {len(value)} chars]")
            else:
                print(f"{var}: {value}")
        else:
            print(f"{var}: [not set]")


def main():
    """Run all diagnostics"""
    print("\n")
    print("*" * 80)
    print("GPU DIAGNOSTIC TOOL")
    print("*" * 80)
    print("\n")
    
    # System info
    check_system_info()
    
    # Environment variables
    check_environment_variables()
    
    # TensorFlow
    tf_ok, gpus = check_tensorflow()
    
    # GPU training test
    if tf_ok:
        training_ok = test_gpu_training()
    else:
        training_ok = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if tf_ok and gpus and training_ok:
        print("[OK] GPU is properly configured and working!")
        print("\nYou can now run training scripts with GPU acceleration:")
        print("  cd research_pipeline/comparison")
        print("  python run_comparison.py")
    elif tf_ok and gpus:
        print("[WARNING] GPU detected but training test failed")
        print("This may indicate a CUDA/cuDNN configuration issue")
        print("\nTroubleshooting:")
        print("1. Check NVIDIA drivers: nvidia-smi")
        print("2. Verify CUDA/cuDNN installation")
        print("3. Try reinstalling TensorFlow: pip install tensorflow --upgrade")
    elif tf_ok and not gpus:
        print("[INFO] TensorFlow is installed but no GPU detected")
        print("\nPossible reasons:")
        print("1. NVIDIA GPU not detected")
        print("2. NVIDIA drivers not installed")
        print("3. CUDA not installed")
        print("4. TensorFlow not built with CUDA support")
        print("\nTo enable GPU:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA toolkit")
        print("3. Install cuDNN")
        print("4. Reinstall TensorFlow: pip install tensorflow --upgrade")
    else:
        print("[ERROR] TensorFlow is not properly installed")
        print("\nTo fix:")
        print("  pip install tensorflow --upgrade")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
