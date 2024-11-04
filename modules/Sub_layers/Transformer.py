import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dim_ffn, dropout=0.1):
        """
        Initializes the Transformer model for multivariate time series.

        Args:
            input_dim (int): Number of features in the input time series (number of variables).
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dim_ffn (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model

        # Linear layer to project input to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_ffn, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to project the Transformer output to the desired output dimension
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_dim)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply Transformer encoder
        x = self.transformer_encoder(x)

        # Project back to original input dimension
        x = self.output_layer(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Adds positional encoding to the input tensor.

        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on position and dimension
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register 'pe' as a buffer to avoid backpropagation through this parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            torch.Tensor: Positional encoded tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

if __name__ == "__main__":
    # Example parameters
    input_dim = 20    # Number of features in the time series
    d_model = 64      # Model dimension
    n_heads = 4       # Number of attention heads
    num_layers = 2    # Number of encoder layers
    dim_ffn = 128     # Dimension of feed-forward network

    # Model initialization
    model = TimeSeriesTransformer(input_dim, d_model, n_heads, num_layers, dim_ffn)

    # Example input (batch_size=32, seq_len=50, input_dim=10)
    x = torch.randn(32, 500, input_dim)

    # Forward pass
    output = model(x)
    print(output.shape)  # Should be (32, 50, input_dim)
