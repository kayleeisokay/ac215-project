"""
DoseWise: Counterfactual Time Series Prediction Model
Encoder-Decoder architecture with dual branches for control vs treatment predictions
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DoseWiseEncoder(nn.Module):
    """
    Encoder module with Conv1D layers, positional encoding, and Transformer encoder
    """
    
    def __init__(
        self, 
        input_features=5,  # ABP, PLETH, CO2, ECG, u
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        conv_channels=[32, 64],
        kernel_size=3
    ):
        super(DoseWiseEncoder, self).__init__()
        
        # Conv1D layers for feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_features,
            out_channels=conv_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(conv_channels[0])
        self.batch_norm2 = nn.BatchNorm1d(conv_channels[1])
        
        # Project conv output to d_model dimension
        self.projection = nn.Linear(conv_channels[1], d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_features]
        Returns:
            hidden_state: Encoded representation [batch_size, seq_len, d_model]
        """
        # Transpose for Conv1D: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Conv layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        # Transpose back: [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # Project to d_model
        x = self.projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        hidden_state = self.transformer_encoder(x)
        
        return hidden_state


class DoseWiseDecoder(nn.Module):
    """
    Decoder module with Transformer decoder and Conv layers
    """
    
    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        conv_channels=[64, 32],
        kernel_size=3,
        output_features=4  # ABP predictions (mean, std, etc.)
    ):
        super(DoseWiseDecoder, self).__init__()
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Conv layers for output refinement
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=conv_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(conv_channels[0])
        self.batch_norm2 = nn.BatchNorm1d(conv_channels[1])
        
        # Final projection to output features
        self.output_projection = nn.Linear(conv_channels[1], output_features)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: Target sequence [batch_size, tgt_seq_len, d_model]
            memory: Encoder hidden state [batch_size, src_seq_len, d_model]
            tgt_mask: Mask for target sequence (for autoregressive decoding)
            memory_mask: Mask for encoder outputs
        Returns:
            output: Predictions [batch_size, tgt_seq_len, output_features]
        """
        # Transformer decoding
        x = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # Transpose for Conv1D: [batch, d_model, seq_len]
        x = x.transpose(1, 2)
        
        # Conv layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        # Transpose back: [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class DoseWiseModel(nn.Module):
    """
    Complete DoseWise model with encoder and dual decoder branches
    for counterfactual prediction (with and without drug intervention)
    """
    
    def __init__(
        self,
        input_features=5,  # ABP, PLETH, CO2, ECG, u
        output_features=4,  # ABP predictions
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        encoder_conv_channels=[32, 64],
        decoder_conv_channels=[64, 32],
        kernel_size=3
    ):
        super(DoseWiseModel, self).__init__()
        
        # Shared encoder
        self.encoder = DoseWiseEncoder(
            input_features=input_features,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            conv_channels=encoder_conv_channels,
            kernel_size=kernel_size
        )
        
        # Control branch decoder (u=0, no drug)
        self.control_decoder = DoseWiseDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            conv_channels=decoder_conv_channels,
            kernel_size=kernel_size,
            output_features=output_features
        )
        
        # Treatment branch decoder (u=dose)
        self.treatment_decoder = DoseWiseDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            conv_channels=decoder_conv_channels,
            kernel_size=kernel_size,
            output_features=output_features
        )
        
        # Learnable query embeddings for decoder input
        self.control_query_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.treatment_query_embed = nn.Parameter(torch.randn(1, 1, d_model))
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive decoding"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x, target_len=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_features]
               Features: ABP, PLETH, CO2, ECG, u (drug dosage)
            target_len: Length of prediction sequence (if None, uses input length)
        Returns:
            control_pred: Predictions without drug [batch_size, target_len, output_features]
            treatment_pred: Predictions with drug [batch_size, target_len, output_features]
        """
        batch_size = x.size(0)
        if target_len is None:
            target_len = x.size(1)
        
        # Encode input sequence
        memory = self.encoder(x)
        
        # Prepare decoder queries
        # Repeat query embeddings for batch and sequence length
        control_tgt = self.control_query_embed.repeat(batch_size, target_len, 1)
        treatment_tgt = self.treatment_query_embed.repeat(batch_size, target_len, 1)
        
        # Generate causal mask for autoregressive decoding
        tgt_mask = self.generate_square_subsequent_mask(target_len).to(x.device)
        
        # Decode for control branch (u=0)
        control_pred = self.control_decoder(
            tgt=control_tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # Decode for treatment branch (u=dose)
        treatment_pred = self.treatment_decoder(
            tgt=treatment_tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        return control_pred, treatment_pred


def create_model(config=None):
    """
    Factory function to create DoseWise model with default or custom config
    
    Args:
        config: Dictionary with model hyperparameters (optional)
    Returns:
        model: DoseWiseModel instance
    """
    default_config = {
        'input_features': 5,
        'output_features': 4,
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'encoder_conv_channels': [32, 64],
        'decoder_conv_channels': [64, 32],
        'kernel_size': 3
    }
    
    if config:
        default_config.update(config)
    
    model = DoseWiseModel(**default_config)
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing DoseWise Model...")
    
    # Create model
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy input
    batch_size = 4
    seq_len = 100
    input_features = 5  # ABP, PLETH, CO2, ECG, u
    
    dummy_input = torch.randn(batch_size, seq_len, input_features)
    
    # Forward pass
    with torch.no_grad():
        control_pred, treatment_pred = model(dummy_input, target_len=50)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Control predictions shape: {control_pred.shape}")
    print(f"Treatment predictions shape: {treatment_pred.shape}")
    print("\nModel test successful!")

