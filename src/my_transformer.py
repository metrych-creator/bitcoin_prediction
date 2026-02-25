import os
import sys
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.pipeline import Pipeline
# Set up project paths for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.data_processor import count_days_since_last_candle, get_crypto_data_yahoo
from src.transformer_architecture import BitcoinPredictor, BitcoinTransformer
from src.transformer_data_processor import prepare_joined_data
from src.transformer_results import show_results

from src.config import COLUMN_TO_PREDICT, HORIZON, WINDOW, BATCH_SIZE
from src.pipeline_tasks import (SlidingWindowDataset)
from src.transformer_architecture import transformer_pipeline


device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def run_production_inference(if_train: bool=False, model_path: str=None, pipeline_path: str=None, epochs: int=5, lr: float=0.0001):
    full_df, processed_df_raw = prepare_joined_data()
    predictor = BitcoinPredictor(input_dim=17+HORIZON, horizon=HORIZON) # input 17 engineered features + horizon target columns????
    train_df = processed_df_raw.dropna()

    target_cols = [col for col in train_df.columns if col.startswith('target_t+')]
    feature_cols = [col for col in train_df.columns if col not in target_cols 
                    and col not in ['Open', 'High', 'Low', 'Close', 'Volume']] # Exclude raw prices

    # --- 2. DATASET & LOADING ---
    ds = SlidingWindowDataset(train_df[feature_cols + target_cols], training_window_size=WINDOW, horizon_size=HORIZON, feature_cols=feature_cols)

    train_size = int(0.8 * len(ds))
    train_ds = Subset(ds, range(train_size))
    test_ds = Subset(ds, range(train_size, len(ds)))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. MODEL INITIALIZATION ---
    model = BitcoinTransformer(input_dim=len(feature_cols), horizon=HORIZON).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(epochs=epochs, verbose=True):
    # --- 4. TRAINING LOOP ---
        print(f"Starting training on {device}...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
                
                optimizer.zero_grad()
                output = model(batch_x)
                
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Avg MSE Loss: {total_loss/len(train_loader):.6f}")


    if if_train:
        train(epochs=5, verbose=True)
        # Save the trained model and pipeline for future inference
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        joblib.dump(transformer_pipeline, pipeline_path)
        print(f"Model and pipeline saved to {model_path} and {pipeline_path}")
    else:
        # Load the trained model and pipeline
        if model_path and pipeline_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print(f"Model loaded from {model_path}")
        else:
            print("Model path or pipeline path not provided. Exiting inference.")
            return


    # --- 5. PREDICT ---
    model.eval()
    with torch.no_grad():
        # Prepare the most recent window for inference
        last_window_df = processed_df_raw.drop(columns=target_cols).tail(WINDOW)
        last_window_df = last_window_df[feature_cols] 

        last_window_scaled = ds.scaler_x.transform(last_window_df.values)
        input_tensor = torch.FloatTensor(last_window_scaled).unsqueeze(0).to(device)
        
        pred_scaled = model(input_tensor)
        # Inverse predictions log-returns
        pred_log_returns = ds.scaler_y.inverse_transform(pred_scaled.cpu().numpy()).flatten()


    # --- 6. RESULTS & PRICE CONVERSION ---
    show_results(full_df, processed_df_raw, pred_log_returns, show_plot=False)

    return pred_log_returns

# run_production_inference(if_train=False, model_path=f'models/transformer/{COLUMN_TO_PREDICT}/model.pkl', pipeline_path=f'models/transformer/{COLUMN_TO_PREDICT}/feature_pipeline.pkl', epochs=5, lr=0.0001)
