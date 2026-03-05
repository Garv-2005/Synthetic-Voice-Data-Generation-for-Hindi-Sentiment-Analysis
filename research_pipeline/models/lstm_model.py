"""
LSTM Model for Speech Emotion Recognition
Bidirectional LSTM for temporal sequence modeling
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.regularizers import l2


class LSTMModel:
    """Bidirectional LSTM model for speech emotion recognition"""
    
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
        self.model_name = "LSTM"
        
    def build_model(self):
        """Build Bidirectional LSTM model architecture"""
        # Reshape input: (mel_bins, time_frames) -> (time_frames, mel_bins)
        # Input shape is (128, 174, 1) -> need (174, 128)
        model = Sequential([
            Reshape((self.input_shape[1], self.input_shape[0]), 
                   input_shape=self.input_shape),
            
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                             kernel_regularizer=l2(self.l2_reg))),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(self.lstm_units // 2, return_sequences=False,
                             kernel_regularizer=l2(self.l2_reg))),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.2),
            
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
