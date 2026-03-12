import pandas as pd
from torch import nn
import numpy as np
from pathlib import Path
import sys

import torch

from src.data_processor import create_datasets, prepare_train_test_data
from src.utils.tools import load_best_params, save_model_and_pipeline
# Set up project paths for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))   
from src.config import HORIZON, TRAINING_PARAMS
from src.pipeline_tasks import PositionalEncoding
from torch.utils.data import DataLoader
from src.utils.logger_config import logger

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


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
    

    def predict(self, last_window: pd.DataFrame):
        """Predicts based on last_window, returns scaled predictions."""
        input_tensor = torch.FloatTensor(last_window).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            pred_scaled = self(input_tensor).to(device)
        return pred_scaled

    def run_training_loop(self, train_loader, epochs: int = None, lr: float = None, verbose: bool = True):
        # Use centralized training parameters or defaults
        epochs = epochs or TRAINING_PARAMS['epochs']
        lr = lr or TRAINING_PARAMS['learning_rate']
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        if verbose:
            logger.info(f"Training model for {epochs} epochs with learning rate {lr} on device {device}")
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = self(batch_x)
                
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Avg MSE Loss: {total_loss/len(train_loader):.6f}")


    def train_with_best_params(self):
        train, test, feature_cols, target_cols, pipeline = prepare_train_test_data()
        
        best_params = load_best_params()

        train_ds, _ = create_datasets(train, test, feature_cols, target_cols)
        train_loader = DataLoader(train_ds, batch_size=TRAINING_PARAMS['batch_size'], shuffle=False)

        self.run_training_loop( 
        train_loader, 
        epochs=best_params.get['epochs'], 
        lr=best_params['learning_rate'], 
        verbose=True
        )

        save_model_and_pipeline(self, pipeline)
        logger.info("Training complete. Model and Pipeline are synced and saved.")

        return pipeline