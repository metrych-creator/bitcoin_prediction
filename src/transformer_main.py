import os
import sys
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
# Set up project paths for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.transformer_architecture import BitcoinTransformer
from src.transformer_data_processor import prepare_joined_data
from src.transformer_results import show_results
from src.data_processor import inverse_transform_predictions

from src.config import COLUMN_TO_PREDICT, HORIZON, WINDOW, BATCH_SIZE
from src.pipeline_tasks import (SlidingWindowDataset)
from src.transformer_architecture import transformer_pipeline


device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))




def train_model(model, train_loader, epochs: int = 5, lr: float = 0.0001, verbose: bool = True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
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
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg MSE Loss: {total_loss/len(train_loader):.6f}")


def save_model_and_pipeline(model, model_path: str, pipeline_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(transformer_pipeline, pipeline_path)
    print(f"Model and pipeline saved to {model_path} and {pipeline_path}")


def load_model_and_pipeline(model, model_path: str, pipeline_path: str):
    if model_path and pipeline_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"Model loaded from {model_path}")
        return True
    else:
        print("Model path or pipeline path not provided. Exiting inference.")
        return False


def predict_with_model(model, processed_df_raw, target_cols, feature_cols, ds):
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
    
    return pred_log_returns


def run_transformer_inference(if_train: bool=False, model_path: str=f'models/transformer/{COLUMN_TO_PREDICT}/model.pkl', pipeline_path: str=f'models/transformer/{COLUMN_TO_PREDICT}/feature_pipeline.pkl', epochs: int=5, lr: float=0.0001):
    """Main function to run production inference or training."""
    full_df, processed_df_raw = prepare_joined_data()
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

    if if_train:
        train_model(model, train_loader, epochs, lr, verbose=True)
        save_model_and_pipeline(model, model_path, pipeline_path)
    else:
        if not load_model_and_pipeline(model, model_path, pipeline_path):
            return

    # --- 5. PREDICT ---
    pred_log_returns = predict_with_model(model, processed_df_raw, target_cols, feature_cols, ds)

    # --- 6. INVERSE TRANSFORM ---
    # Get the last HORIZON prices for inverse transformation based on COLUMN_TO_PREDICT
    from src.config import COLUMN_TO_PREDICT
    
    if COLUMN_TO_PREDICT == 'Close_log_return':
        last_prices = processed_df_raw['Close'].tail(HORIZON)
    else:  # For volatility or other targets
        last_prices = processed_df_raw[COLUMN_TO_PREDICT.replace('_log_return', '')].tail(HORIZON)
    
    pred_prices = inverse_transform_predictions(pred_log_returns, last_prices)

    # --- 7. RESULTS & PRICE CONVERSION ---
    show_results(full_df, processed_df_raw, pred_log_returns, pred_prices, show_plot=True)

    return pred_log_returns, pred_prices

