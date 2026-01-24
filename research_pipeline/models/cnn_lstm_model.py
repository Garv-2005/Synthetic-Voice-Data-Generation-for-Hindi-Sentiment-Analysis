"""
CNN-LSTM (CRNN) Model for Speech Emotion Recognition
Combines CNN for spatial features and LSTM for temporal modeling
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, LSTM, Bidirectional, Dense, 
    Dropout, BatchNormalization, Reshape, TimeDistributed, Flatten
)
from tensorflow.keras.regularizers import l2


class CNNLSTMModel:
    """CNN-LSTM hybrid model (CRNN) for speech emotion recognition"""
    
    def __init__(self, input_shape, num_classes, lstm_units=128, dropout_rate=0.5, l2_reg=0.0001):
        """
        Args:
            input_shape: Shape of input features (excluding batch dimension)
            num_classes: Number of emotion classes
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization parameter
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        self.model_name = "CNN-LSTM"
        
    def build_model(self):
        """Build CNN-LSTM hybrid model architecture"""
        from tensorflow.keras.layers import Input
        
        inputs = Input(shape=self.input_shape)
        
        # CNN layers for feature extraction
        x = Conv2D(32, (3, 3), activation='relu', 
                  kernel_regularizer=l2(self.l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(64, (3, 3), activation='relu',
                  kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(128, (3, 3), activation='relu',
                  kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Reshape for LSTM: (batch, time, features)
        # After pooling, shape is (batch, height, width, channels)
        # We treat height as time dimension and width*channels as features
        x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
        
        # LSTM layers for temporal modeling
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                             kernel_regularizer=l2(self.l2_reg)))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=False,
                             kernel_regularizer=l2(self.l2_reg)))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
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
