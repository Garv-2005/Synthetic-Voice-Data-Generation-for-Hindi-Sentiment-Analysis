"""
Generic model training script
Can train any model architecture from the models directory
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

# TensorFlow imports (only needed for deep learning models)
_GPU_CONFIGURED = False  # Flag to ensure GPU message prints only once

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
    
    # Configure GPU settings (only print message once)
    if not _GPU_CONFIGURED:
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
            print("[CPU] To use GPU, ensure CUDA and cuDNN are installed.")
        _GPU_CONFIGURED = True
    
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    Callback = object
    if not _GPU_CONFIGURED:
        print("⚠ TensorFlow not available. Deep learning models cannot be trained.")
        _GPU_CONFIGURED = True

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


if TF_AVAILABLE:
    class ProgressBarCallback(Callback):
        """Progress bar callback for TensorFlow/Keras training"""
        def __init__(self, epochs, model_name):
            super().__init__()
            self.epochs = epochs
            self.model_name = model_name
            self.pbar = None
            
        def on_train_begin(self, logs=None):
            self.pbar = tqdm(total=self.epochs, desc=f'{self.model_name} Training', 
                            unit='epoch', ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            
            # Update progress bar with metrics
            acc = logs.get('accuracy', 0)
            val_acc = logs.get('val_accuracy', 0)
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            
            self.pbar.set_postfix({
                'acc': f'{acc:.3f}',
                'val_acc': f'{val_acc:.3f}',
                'loss': f'{loss:.3f}',
                'val_loss': f'{val_loss:.3f}'
            })
            self.pbar.update(1)
            
        def on_train_end(self, logs=None):
            if self.pbar is not None:
                self.pbar.close()
else:
    ProgressBarCallback = None


class F1Callback(Callback):
    """Callback to compute F1 score during training (for TensorFlow models)"""
    def __init__(self, X_val, y_val, X_train=None, y_train=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.X_train = X_train
        self.y_train = y_train
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        # Validation F1
        y_pred_probs = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        val_f1 = f1_score(self.y_val, y_pred, average='macro', zero_division=0)
        logs['val_f1'] = val_f1
        
        # Training F1 (on a sample if training data provided)
        if self.X_train is not None and self.y_train is not None:
            sample_size = min(500, len(self.X_train))
            indices = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_train_sample = self.X_train[indices]
            y_train_sample = self.y_train[indices]
            
            y_train_pred_probs = self.model.predict(X_train_sample, verbose=0)
            y_train_pred = np.argmax(y_train_pred_probs, axis=1)
            train_f1 = f1_score(y_train_sample, y_train_pred, average='macro', zero_division=0)
            logs['f1'] = train_f1


def train_model(model_class, data_path, results_dir, model_name,
                epochs=100, batch_size=32, random_seed=42, skip_if_exists=False):
    """
    Train a model with comprehensive evaluation
    
    Args:
        model_class: Model class to instantiate
        data_path: Path to .npz file with features
        results_dir: Directory to save results
        model_name: Name of the model
        epochs: Number of training epochs
        batch_size: Batch size
        random_seed: Random seed for reproducibility
    """
    # Set random seeds
    np.random.seed(random_seed)
    if TF_AVAILABLE:
        tf.random.set_seed(random_seed)
    
    print("=" * 60)
    print(f"{model_name.upper()} MODEL TRAINING")
    print("=" * 60)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if it's an SVM model (needed for skip check)
    is_svm = model_name.upper() == 'SVM'
    
    # Check if model already exists and skip if requested
    if skip_if_exists:
        model_ext = '.pkl' if is_svm else '.h5'
        model_path = os.path.join(results_dir, f'best_{model_name.lower()}_model{model_ext}')
        results_path = os.path.join(results_dir, f'{model_name.lower()}_results.json')
        
        if os.path.exists(model_path) and os.path.exists(results_path):
            print(f"\n✓ {model_name} model already exists. Loading results...")
            with open(results_path, 'r') as f:
                results = json.load(f)
            print(f"  Previous accuracy: {results['metrics']['overall']['accuracy']:.4f}")
            print(f"  Previous macro F1: {results['metrics']['overall']['macro_f1']:.4f}")
            print(f"  Skipping training. Use --force to retrain.\n")
            return results, None
    
    # Load data
    print("\n[1/7] Loading features...")
    with tqdm(total=1, desc="Loading data", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
        data = np.load(data_path, allow_pickle=True)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        emotion_map = data['emotion_map'].item()
        pbar.update(1)
    
    num_classes = len(emotion_map)
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Feature shape: {X_train.shape[1:]}")
    
    # Normalize features
    print("\n[2/7] Normalizing features...")
    with tqdm(total=3, desc="Normalizing", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        X_train = X_train.astype('float32') / 255.0 if X_train.max() > 1.0 else X_train.astype('float32')
        pbar.update(1)
        X_val = X_val.astype('float32') / 255.0 if X_val.max() > 1.0 else X_val.astype('float32')
        pbar.update(1)
        X_test = X_test.astype('float32') / 255.0 if X_test.max() > 1.0 else X_test.astype('float32')
        pbar.update(1)
    
    # Build model
    print("\n[3/7] Building model...")
    model = model_class(
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )
    
    built_model = model.build_model()
    
    if not is_svm:
        try:
            built_model.summary()
        except:
            pass
    
    # Train model
    print("\n[4/7] Training model...")
    
    if is_svm:
        # SVM training (no callbacks, no epochs)
        print("\n[4/7] Training SVM model...")
        with tqdm(total=1, desc="Training SVM", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            history = model.train(X_train, y_train, X_val, y_val, verbose=0)
            pbar.update(1)
        
        # Evaluate on test set
        print("\n[5/7] Evaluating model...")
        with tqdm(total=1, desc="Evaluating", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            test_loss, test_accuracy = model.evaluate(X_test, y_test)
            pbar.update(1)
        
        # Predictions
        print("  Making predictions...")
        with tqdm(total=1, desc="Predicting", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            pbar.update(1)
        
        # Save SVM model (using joblib, not h5)
        import joblib
        model_path = os.path.join(results_dir, f'best_{model_name.lower()}_model.pkl')
        with tqdm(total=1, desc="Saving model", ncols=100, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            model.save_model(model_path)
            pbar.update(1)
        
    else:
        # Deep learning model training
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for deep learning models")
        
        # Setup callbacks
        model_checkpoint = ModelCheckpoint(
            os.path.join(results_dir, f'best_{model_name.lower()}_model.h5'),
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
        
        # F1 callback
        callbacks = [model_checkpoint, early_stopping, reduce_lr]
        if F1Callback is not None:
            f1_callback = F1Callback(X_val, y_val, X_train, y_train)
            callbacks.append(f1_callback)
        
        # Train model with standard TensorFlow output (verbose=1)
        # This shows detailed epoch-by-epoch progress with batch-level updates
        print("\n[4/7] Training model...")
        print(f"Training {model_name} for {epochs} epochs...\n")
        history = built_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1  # Show standard TensorFlow training output (epoch-by-epoch)
        )
        print()  # Add blank line after training completes
        
        # Evaluate on test set
        print("\n[5/7] Evaluating model...")
        test_loss, test_accuracy = built_model.evaluate(X_test, y_test, verbose=1)
        
        # Predictions
        print("  Making predictions...")
        batch_size_pred = 32
        n_batches = (len(X_test) + batch_size_pred - 1) // batch_size_pred
        y_pred_probs = []
        with tqdm(total=n_batches, desc="Predicting", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for i in range(0, len(X_test), batch_size_pred):
                batch = X_test[i:i+batch_size_pred]
                batch_pred = built_model.predict(batch, verbose=0)
                y_pred_probs.append(batch_pred)
                pbar.update(1)
        y_pred_probs = np.vstack(y_pred_probs)
        y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, emotion_map)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} MODEL RESULTS")
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
    print("\n[6/7] Generating visualizations...")
    
    # Lazy import visualization functions
    from utils.visualization import (
        plot_training_history,
        plot_comprehensive_training_history,
        plot_confusion_matrix,
        plot_per_class_metrics
    )
    
    viz_tasks = []
    
    # Standard training history (skip for SVM as it doesn't have training history)
    if not is_svm and hasattr(history, 'history'):
        viz_tasks.append(('Training history', lambda: plot_training_history(
            history.history,
            save_path=os.path.join(results_dir, 'training_history.png')
        )))
        viz_tasks.append(('Comprehensive history', lambda: plot_comprehensive_training_history(
            history.history,
            save_path=os.path.join(results_dir, 'comprehensive_training_history.png'),
            model_name=model_name
        )))
    
    viz_tasks.extend([
        ('Confusion matrix', lambda: plot_confusion_matrix(
            y_test, y_pred, emotion_map,
            save_path=os.path.join(results_dir, 'confusion_matrix.png')
        )),
        ('Normalized confusion matrix', lambda: plot_confusion_matrix(
            y_test, y_pred, emotion_map,
            save_path=os.path.join(results_dir, 'confusion_matrix_normalized.png'),
            normalize=True
        )),
        ('Per-class metrics', lambda: plot_per_class_metrics(
            y_test, y_pred, emotion_map,
            save_path=os.path.join(results_dir, 'per_class_metrics.png')
        ))
    ])
    
    with tqdm(total=len(viz_tasks), desc="Generating plots", ncols=100, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for task_name, task_func in viz_tasks:
            task_func()
            pbar.set_postfix_str(task_name)
            pbar.update(1)
    
    # Save results
    print("\n[7/7] Saving results...")
    results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'input_shape': list(X_train.shape[1:]),
            'num_classes': num_classes,
            'random_seed': random_seed
        },
        'metrics': metrics
    }
    
    # Add training-specific config
    if not is_svm and hasattr(history, 'history'):
        results['configuration']['epochs'] = len(history.history['loss'])
        results['configuration']['batch_size'] = batch_size
        results['training_history'] = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        # Add F1 scores if available
        if 'f1' in history.history:
            results['training_history']['f1'] = [float(x) for x in history.history['f1']]
            results['training_history']['val_f1'] = [float(x) for x in history.history['val_f1']]
    else:
        # SVM doesn't have training history
        results['configuration']['kernel'] = model.kernel if hasattr(model, 'kernel') else 'rbf'
        results['configuration']['C'] = model.C if hasattr(model, 'C') else 1.0
        results['training_history'] = {
            'note': 'SVM does not have iterative training history'
        }
    
    results_path = os.path.join(results_dir, f'{model_name.lower()}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    if is_svm:
        print(f"Model saved to: {os.path.join(results_dir, f'best_{model_name.lower()}_model.pkl')}")
    else:
        print(f"Model saved to: {os.path.join(results_dir, f'best_{model_name.lower()}_model.h5')}")
    
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} TRAINING COMPLETE")
    print("=" * 60)
    
    return results, built_model
