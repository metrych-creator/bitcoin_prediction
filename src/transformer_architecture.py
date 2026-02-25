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
from src.config import COLUMN_TO_PREDICT, HORIZON
from src.data_processor import count_days_since_last_candle, get_crypto_data_yahoo
from src.pipeline_tasks import DiffTransformer, FeatureEngineer, LogTransformer, PositionalEncoding, TechnicalFeaturesAdder, TimeSeriesShifter, TimeSeriesImputer


class BitcoinTransformer(nn.Module):
    """
    Transformer Encoder based architecture for Time Series forecasting.
    Maps a window of historical features to a vector of future log-returns.
    """
    def __init__(self, input_dim, model_dim=128, n_heads=8, n_layers=3, horizon=1, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        # Project raw features into the model dimension
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        
        # Define the Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dropout=dropout, batch_first=True
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

    
transformer_pipeline = Pipeline([
        # ('date_formatter', DateFormatter(column_name='Date')),
        ('imputer', TimeSeriesImputer(freq='D')), # Fill missing dates
        ('tech_features', TechnicalFeaturesAdder()), # RSI, Bollinger Bands
        ('log_transformer', LogTransformer(columns=['Open', 'High', 'Low', 'Close', 'Volume'])),
        ('diff_transformer', DiffTransformer(degree=1, verbose=False)), # Make series stationary
        ('feature_engineer', FeatureEngineer()), # MAs, Volatility, Lags
        ('shifter', TimeSeriesShifter(horizon=HORIZON, target_col=COLUMN_TO_PREDICT))
    ])