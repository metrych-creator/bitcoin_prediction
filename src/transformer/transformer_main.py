from itertools import product
import json
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

from src.transformer.transformer_pipeline import transformer_pipeline
from src.utils.logger_config import logger
from src.utils.tools import load_model_and_pipeline, save_best_params, save_model_and_pipeline
from src.transformer.transformer_architecture import BitcoinTransformer
from src.transformer.transformer_data_processor import prepare_joined_data
from src.data_processor import inverse_scale_targets, inverse_transform_predictions
from src.config import HORIZON, HYPERPARAMETER_GRIDS, WINDOW, BATCH_SIZE, TRAINING_PARAMS
from src.config_manager import get_config
from src.pipeline_tasks import SlidingWindowDataset

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def train_model(model, train_loader, epochs: int = None, lr: float = None, verbose: bool = True):
    # Use centralized training parameters or defaults
    epochs = epochs or TRAINING_PARAMS['epochs']
    lr = lr or TRAINING_PARAMS['learning_rate']
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
        logger.info(f"Training model for {epochs} epochs with learning rate {lr} on device {device}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Avg MSE Loss: {total_loss/len(train_loader):.6f}")



def predict_with_model(model, processed_df_raw, pipeline, target_cols, feature_cols, ds):
    model.eval()
    with torch.no_grad():
        # Prepare the most recent window for inference
        last_window = processed_df_raw[feature_cols].tail(WINDOW).values
        input_tensor = torch.FloatTensor(last_window).unsqueeze(0).to(device)

        # prediction
        pred_scaled = model(input_tensor).to(device)
        pred_log_returns = inverse_scale_targets(pred_scaled, pipeline, HORIZON)
    return pred_log_returns


def train_with_grid_search(model_path: str=None, pipeline_path: str=None, verbose: bool=True):
    """
    Train models with different hyperparameter combinations from a grid.
    
    Args:
        grid_params: Dictionary containing hyperparameter grids:
            - learning_rate: List of learning rates to try
            - epochs: List of epoch counts to try
            - batch_sizes: List of batch sizes to try
        model_path: Path to save the best model
        pipeline_path: Path to save the feature pipeline
        verbose: Whether to print training progress
    
    Returns:
        best_model: The model with the best validation performance
        best_params: The hyperparameters that produced the best model
        results: Dictionary with training results for all parameter combinations
    """
    grid_params = HYPERPARAMETER_GRIDS['Transformer']
    config = get_config()
    window_size = config.get_window_size()
    
    # grid params
    keys = grid_params.keys()
    values = grid_params.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Prepare data
    full_df = prepare_joined_data()
    
    # Train/Test Split
    split_idx = int(len(full_df) * 0.7)
    train_raw = full_df.iloc[:split_idx]
    test_raw = full_df.iloc[split_idx:]

    # Apply transformations
    train_processed = transformer_pipeline.fit_transform(train_raw)
    test_processed = transformer_pipeline.transform(test_raw)

    # Features/Targets Split
    target_cols = [col for col in train_processed.columns if col.startswith('target_t+')]
    feature_cols = [col for col in train_processed.columns if col not in target_cols 
                    and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    

    train_df = train_processed.dropna()
    test_df = test_processed.dropna()


    # --- 2. DATASET & LOADING ---
    train_ds = SlidingWindowDataset(train_df[feature_cols + target_cols], training_window_size=window_size, horizon_size=HORIZON, feature_cols=feature_cols)
    test_ds = SlidingWindowDataset(test_df[feature_cols + target_cols], training_window_size=window_size, horizon_size=HORIZON, feature_cols=feature_cols)


    # Initialize results storage
    results = []
    best_model = None
    best_score = float('inf')
    best_params = None

    for params in combinations:
        # Extract parameters (use defaults from config if not in grid)
        lr = params.get('learning_rate', TRAINING_PARAMS['learning_rate'])
        epochs = params.get('epochs', TRAINING_PARAMS['epochs'])
        batch_size = params.get('batch_size', BATCH_SIZE)
            
        # 2. Initialize Model with injected architecture params
        model = BitcoinTransformer(
            input_dim=len(feature_cols),
            model_dim=params.get('model_dim', 64),
            n_heads=params.get('n_heads', 4),
            n_layers=params.get('n_layers', 2),
            dropout=params.get('dropout', 0.1)
        ).to(device)

        # 3. Create loaders for this batch size
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        if verbose:
            logger.info(f"Testing combination: {params}")

        # 4. Train
        train_model(model, train_loader, epochs=epochs, lr=lr, verbose=verbose)

        # 5. Evaluate
        # evalueate_model()
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                loss = criterion(model(batch_x), batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        params['validation_loss'] = avg_loss
        results.append(params)

        if verbose:
            logger.info(f"Validation Loss: {avg_loss:.6f}")

        # 6. Track Best
        if avg_loss < best_score:
            best_score = avg_loss
            best_params = params
            best_model = model

    # Save Best
    if best_model is not None:
        save_model_and_pipeline(best_model, transformer_pipeline)
        save_best_params(best_params)

    return best_model, best_params, results


def run_transformer_inference(if_train: bool=False, model_path: str=None, pipeline_path: str=None, epochs: int=5, lr: float=0.0001):
    """Main function to run production inference or training."""
    
    logger.debug(f"Running transformer inference with if_train={if_train}, model_path={model_path}, pipeline_path={pipeline_path}, epochs={epochs}, lr={lr}")
    
    # Get dynamic configuration
    config = get_config()
    window_size = config.get_window_size()
    column_to_predict = config.get_column_to_predict()
    
    # Set default paths if not provided
    if model_path is None:
        model_path = f'models/transformer/{column_to_predict}/model.pkl'
        if not if_train and not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found. Please train the model first.")
            return None, None
        
    if pipeline_path is None:
        pipeline_path = f'models/transformer/{column_to_predict}/feature_pipeline.pkl'
        if not if_train and not os.path.exists(pipeline_path):
            logger.warning(f"Pipeline file {pipeline_path} not found. Please train the model first.")
            return None, None
    
    full_df, processed_df_raw = prepare_joined_data()
    train_df = processed_df_raw.dropna()

    target_cols = [col for col in train_df.columns if col.startswith('target_t+')]
    feature_cols = [col for col in train_df.columns if col not in target_cols 
                    and col not in ['Open', 'High', 'Low', 'Close', 'Volume']] # Exclude raw prices

    # --- 2. DATASET & LOADING ---
    ds = SlidingWindowDataset(train_df[feature_cols + target_cols], training_window_size=window_size, horizon_size=HORIZON, feature_cols=feature_cols)

    train_size = int(0.7 * len(ds))
    train_ds = Subset(ds, range(train_size))
    test_ds = Subset(ds, range(train_size, len(ds)))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    logger.debug(f"Dataset prepared with {len(train_ds)} training samples and {len(test_ds)} testing samples.")

    # --- 3. MODEL INITIALIZATION ---
    model = BitcoinTransformer(input_dim=len(feature_cols), horizon=HORIZON).to(device)

    if if_train:
        train_model(model, train_loader, epochs, lr, verbose=True)
        save_model_and_pipeline(model, model_path, pipeline_path)
    else:
        pipeline = load_model_and_pipeline(model, model_path, pipeline_path)
        if pipeline is None:
            logger.error("Inference aborted due to missing model or pipeline.")
            return None, None

    # --- 5. PREDICT ---
    pred_log_returns = predict_with_model(model, processed_df_raw, pipeline, target_cols, feature_cols, ds)

    # --- 6. INVERSE TRANSFORM ---
    # Get the last HORIZON prices for inverse transformation based on COLUMN_TO_PREDICT
    if column_to_predict == 'Close_log_return':
        last_prices = processed_df_raw['Close'].tail(HORIZON)
    else:  # For volatility or other targets
        last_prices = processed_df_raw[column_to_predict.replace('_log_return', '')].tail(HORIZON)
    
    pred_prices = inverse_transform_predictions(pred_log_returns, last_prices)

    return pred_log_returns, pred_prices



