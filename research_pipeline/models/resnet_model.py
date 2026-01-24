"""
ResNet-style CNN Model for Speech Emotion Recognition
Residual connections for deeper networks
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Add, GlobalAveragePooling2D, Dense, Dropout, Input
)
from tensorflow.keras.regularizers import l2


class ResNetModel:
    """ResNet-style CNN with residual connections"""
    
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
        self.model_name = "ResNet"
        
    def _residual_block(self, x, filters, kernel_size=3, stride=1):
        """Residual block with two conv layers"""
        shortcut = x
        
        # First conv
        x = Conv2D(filters, kernel_size, strides=stride, padding='same',
                  kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Second conv
        x = Conv2D(filters, kernel_size, strides=1, padding='same',
                  kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        
        # Match dimensions if needed
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                            kernel_regularizer=l2(self.l2_reg))(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # Add residual connection
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        
        return x
        
    def build_model(self):
        """Build ResNet-style model architecture"""
        inputs = Input(shape=self.input_shape)
        
        # Initial conv block
        x = Conv2D(32, (7, 7), strides=2, padding='same',
                  kernel_regularizer=l2(self.l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = self._residual_block(x, 128)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = self._residual_block(x, 256)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Global average pooling
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output Layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
