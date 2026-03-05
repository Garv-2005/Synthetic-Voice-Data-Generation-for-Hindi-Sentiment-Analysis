"""
SVM Model for Speech Emotion Recognition
Support Vector Machine classifier for classical ML baseline
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


class SVMModel:
    """SVM model for speech emotion recognition"""
    
    def __init__(self, input_shape, num_classes, kernel='rbf', C=1.0, gamma='scale'):
        """
        Args:
            input_shape: Shape of input features (will be flattened)
            num_classes: Number of emotion classes
            kernel: SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()
        self.model_name = "SVM"
        
    def build_model(self):
        """Build SVM model"""
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=42,
            probability=True  # Enable probability estimates for metrics
        )
        return self.model
    
    def _flatten_features(self, X):
        """Flatten 2D/3D features to 1D for SVM"""
        if len(X.shape) > 2:
            # Flatten: (samples, height, width, channels) -> (samples, height*width*channels)
            n_samples = X.shape[0]
            X_flat = X.reshape(n_samples, -1)
        else:
            X_flat = X
        return X_flat
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=None, batch_size=None, callbacks=None, verbose=1):
        """
        Train the SVM model
        
        Note: epochs, batch_size, callbacks are ignored for SVM (for API compatibility)
        """
        if self.model is None:
            self.build_model()
        
        # Flatten features
        X_train_flat = self._flatten_features(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        if verbose:
            print(f"Training SVM with {X_train_scaled.shape[0]} samples")
            print(f"Feature dimension: {X_train_scaled.shape[1]}")
        
        # Train SVM
        self.model.fit(X_train_scaled, y_train)
        
        # Create mock history for compatibility with deep learning framework
        history = type('History', (), {
            'history': {
                'accuracy': [1.0],  # SVM doesn't have training history
                'val_accuracy': [1.0],
                'loss': [0.0],
                'val_loss': [0.0]
            }
        })()
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Flatten and scale features
        X_test_flat = self._flatten_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        
        # SVM doesn't have loss in the same sense, return 0
        test_loss = 0.0
        
        return test_loss, accuracy
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Flatten and scale features
        X_flat = self._flatten_features(X)
        X_scaled = self.scaler.transform(X_flat)
        
        # Get probability estimates
        y_pred_probs = self.model.predict_proba(X_scaled)
        
        return y_pred_probs
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
        }
        joblib.dump(model_data, filepath)
    
    @staticmethod
    def load_model(filepath):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        svm_model = SVMModel(
            input_shape=model_data['input_shape'],
            num_classes=model_data['num_classes'],
            kernel=model_data['kernel'],
            C=model_data['C'],
            gamma=model_data['gamma']
        )
        svm_model.model = model_data['model']
        svm_model.scaler = model_data['scaler']
        return svm_model
