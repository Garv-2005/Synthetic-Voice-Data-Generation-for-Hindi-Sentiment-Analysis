"""
Direct training script for Classical augmented data
Bypasses comparison framework to test if training works
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add paths
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))

from models import CNNModel, LSTMModel, CNNLSTMModel, ResNetModel, TransformerModel, SVMModel


def main():
    # Load classical augmented data
    data_path = base_dir / "data" / "classical_augmented_features.npz"
    print(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return
    
    data = np.load(data_path, allow_pickle=True)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    emotion_map = data['emotion_map'].item()
    
    print(f"✓ Loaded data:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Emotions: {emotion_map}")
    
    # Train models
    models_to_train = [
        ('CNN', CNNModel),
        ('LSTM', LSTMModel),
        ('CNN-LSTM', CNNLSTMModel),
        ('ResNet', ResNetModel),
        ('Transformer', TransformerModel),
        ('SVM', SVMModel),
    ]
    
    for model_name, model_class in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = model_class(input_shape=X_train.shape[1:], num_classes=len(emotion_map))
            
            # Scale features for SVM
            if model_name == 'SVM':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
                X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))
                X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))
                
                print(f"Training SVM...")
                model.fit(X_train_scaled, y_train)
                
                print(f"Evaluating...")
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                print(f"✓ {model_name} - Accuracy: {acc:.4f}")
            else:
                # Train deep learning model
                print(f"Training {model_name} for 30 epochs...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=32,
                    verbose=0
                )
                
                # Evaluate
                print(f"Evaluating...")
                y_pred = model.predict(X_test)
                if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                acc = accuracy_score(y_test, y_pred)
                print(f"✓ {model_name} - Accuracy: {acc:.4f}")
                
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All models trained!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
