"""
Transformer-based Model for Speech Emotion Recognition
Multi-head attention mechanism for sequence modeling
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Reshape, Add
)
from tensorflow.keras.regularizers import l2


class TransformerModel:
    """Transformer-based model with multi-head attention"""
    
    def __init__(self, input_shape, num_classes, d_model=128, num_heads=4, 
                 num_layers=2, dropout_rate=0.5, l2_reg=0.0001):
        """
        Args:
            input_shape: Shape of input features (excluding batch dimension)
            num_classes: Number of emotion classes
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization parameter
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        self.model_name = "Transformer"
        
    def _transformer_block(self, x, d_model, num_heads):
        """Transformer block with multi-head attention and feed-forward"""
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads
        )(x, x)
        attn_output = Dropout(self.dropout_rate)(attn_output)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        # Feed-forward network
        ffn_output = Dense(d_model * 2, activation='relu',
                          kernel_regularizer=l2(self.l2_reg))(x)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        ffn_output = Dense(d_model, kernel_regularizer=l2(self.l2_reg))(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization()(x)
        
        return x
        
    def build_model(self):
        """Build Transformer-based model architecture"""
        inputs = Input(shape=self.input_shape)
        
        # Reshape to sequence: (mel_bins, time_frames) -> (time_frames, mel_bins)
        x = Reshape((self.input_shape[1], self.input_shape[0]))(inputs)
        
        # Project to d_model dimension
        x = Dense(self.d_model, kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Stack transformer blocks
        for _ in range(self.num_layers):
            x = self._transformer_block(x, self.d_model, self.num_heads)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
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
