"""
CNN Model for Speech Emotion Recognition
Based on baseline CNN architecture
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


class CNNModel:
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
        self.model_name = "CNN"
        
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
