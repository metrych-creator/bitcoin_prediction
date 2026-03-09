from torch import nn
import pandas as pd
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys
# Set up project paths for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))   
from src.config import COLUMN_TO_PREDICT, HORIZON, HYPERPARAMETER_GRIDS
from src.pipeline_tasks import PositionalEncoding


class BitcoinTransformer(nn.Module):
    """
    Transformer Encoder based architecture for Time Series forecasting.
    Maps a window of historical features to a vector of future log-returns.
    """
    def __init__(self, input_dim: int, model_dim: int=128, n_heads: int=8, n_layers: int=3, dropout: float=0.1, horizon: int=HORIZON):
        super().__init__()
        # Use centralized hyperparameters or defaults
        self.model_dim = model_dim
        self.n_layers = n_layers
        
        # Project raw features into the model dimension
        self.input_projection = nn.Linear(input_dim, model_dim)
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(model_dim)
        
        # Define the Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # Final linear layer to output prediction for the specified horizon
        self.decoder = nn.Linear(model_dim, horizon)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(x) * np.sqrt(self.model_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Take the output of the last time step only (Many-to-One)
        return self.decoder(x[:, -1, :])

