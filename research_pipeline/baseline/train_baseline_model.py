"""
Baseline Model Training for Hindi Speech Emotion Recognition
Trains CNN model on baseline features with comprehensive evaluation
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
import tensorflow as tf

# Configure GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] {len(gpus)} GPU(s) available")
        print(f"[GPU] Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"[GPU ERROR] {e}")
else:
    print("[CPU] No GPU detected. Training will use CPU (slower).")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import (
    plot_training_history, 
    plot_confusion_matrix, 
    plot_per_class_metrics
)


class BaselineModel:
    """Baseline CNN model for speech emotion recognition"""
    
    def __init__(self, input_shape, num_classes, dropout_rate=0.5, l2_reg=0.0001):
        """
        Args:
            input_shape: Shape of input features (excluding batch dimension)
            num_classes: Number of emotion classes
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization parameter
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
    def build_model(self):
        """Build CNN model architecture"""
        model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', 
                  input_shape=self.input_shape,
                  kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu',
                  kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu',
                  kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Conv Block
            Conv2D(256, (3, 3), activation='relu',
                  kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output Layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, 
              callbacks=None, verbose=1):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        if callbacks is None:
            callbacks = []
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return test_loss, test_accuracy
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)


def calculate_metrics(y_true, y_pred, emotion_map):
    """Calculate comprehensive evaluation metrics"""
    emotion_names = {v: k for k, v in emotion_map.items()}
    sorted_labels = sorted(emotion_map.values())
    label_names = [emotion_names[i] for i in sorted_labels]
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    classification_rep = classification_report(
        y_true, y_pred, 
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=sorted_labels)
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1)
        },
        'per_class': classification_rep,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def train_baseline_model(data_path, results_dir='../results/baseline',
                        epochs=100, batch_size=32, random_seed=42):
    """
    Train baseline model with comprehensive evaluation
    
    Args:
        data_path: Path to .npz file with features
        results_dir: Directory to save results
        epochs: Number of training epochs
        batch_size: Batch size
        random_seed: Random seed for reproducibility
    """
    # Set random seeds
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading features...")
    data = np.load(data_path, allow_pickle=True)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    emotion_map = data['emotion_map'].item()
    
    num_classes = len(emotion_map)
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Feature shape: {X_train.shape[1:]}")
    
    # Normalize features
    print("\n[2/6] Normalizing features...")
    X_train = X_train.astype('float32') / 255.0 if X_train.max() > 1.0 else X_train.astype('float32')
    X_val = X_val.astype('float32') / 255.0 if X_val.max() > 1.0 else X_val.astype('float32')
    X_test = X_test.astype('float32') / 255.0 if X_test.max() > 1.0 else X_test.astype('float32')
    
    # Build model
    print("\n[3/6] Building model...")
    model = BaselineModel(
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )
    
    baseline_model = model.build_model()
    baseline_model.summary()
    
    # Setup callbacks
    model_checkpoint = ModelCheckpoint(
        os.path.join(results_dir, 'best_baseline_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [model_checkpoint, early_stopping, reduce_lr]
    
    # Train model
    print("\n[4/6] Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Evaluate on test set
    print("\n[5/6] Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, emotion_map)
    
    # Print results
    print("\n" + "=" * 60)
    print("BASELINE MODEL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Macro F1-Score: {metrics['overall']['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['overall']['weighted_f1']:.4f}")
    
    print("\nPer-Class Performance:")
    emotion_names = {v: k for k, v in emotion_map.items()}
    for label_id in sorted(emotion_map.values()):
        emotion_name = emotion_names[label_id]
        if emotion_name in metrics['per_class']:
            prec = metrics['per_class'][emotion_name]['precision']
            rec = metrics['per_class'][emotion_name]['recall']
            f1 = metrics['per_class'][emotion_name]['f1-score']
            print(f"  {emotion_name:15s}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    
    # Training history
    plot_training_history(
        history.history,
        save_path=os.path.join(results_dir, 'training_history.png')
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred, emotion_map,
        save_path=os.path.join(results_dir, 'confusion_matrix.png')
    )
    
    # Normalized confusion matrix
    plot_confusion_matrix(
        y_test, y_pred, emotion_map,
        save_path=os.path.join(results_dir, 'confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # Per-class metrics
    plot_per_class_metrics(
        y_test, y_pred, emotion_map,
        save_path=os.path.join(results_dir, 'per_class_metrics.png')
    )
    
    # Save results
    results = {
        'model_name': 'Baseline CNN',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'input_shape': list(X_train.shape[1:]),
            'num_classes': num_classes,
            'epochs': len(history.history['loss']),
            'batch_size': batch_size,
            'random_seed': random_seed
        },
        'metrics': metrics,
        'training_history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }
    
    results_path = os.path.join(results_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {os.path.join(results_dir, 'best_baseline_model.h5')}")
    
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)
    
    return results, baseline_model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--data_path', type=str, 
                       default='../data/baseline_features.npz',
                       help='Path to feature file')
    parser.add_argument('--results_dir', type=str,
                       default='../results/baseline',
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    train_baseline_model(
        data_path=args.data_path,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
