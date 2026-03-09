from itertools import product
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import time
from datetime import timedelta

# Set up project paths for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.transformer.transformer_pipeline import transformer_pipeline
from src.utils.logger_config import logger
from src.utils.tools import load_best_params, load_model_and_pipeline, save_best_params, save_model_and_pipeline
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
        
        if verbose and (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Avg MSE Loss: {total_loss/len(train_loader):.6f}")



def predict_with_model(model, train_df, feature_cols):
    model.eval()
    with torch.no_grad():
        # Prepare the most recent window for inference
        last_window = train_df[feature_cols].tail(WINDOW).values
        input_tensor = torch.FloatTensor(last_window).unsqueeze(0).to(device)

        # prediction
        pred_scaled = model(input_tensor).to(device)
    return pred_scaled


def _prepare_training_data():
    """Prepare training and test datasets with transformations."""
    # Prepare data and split
    full_df = prepare_joined_data()

    split_idx = int(len(full_df) * 0.7)
    train_raw = full_df.iloc[:split_idx]
    test_raw = full_df.iloc[split_idx:]

    # Transform data
    train_processed = transformer_pipeline.fit_transform(train_raw)

    context_size = 365
    test_with_context = pd.concat([train_raw.tail(context_size), test_raw])
    test_processed_with_context = transformer_pipeline.transform(test_with_context)
    test_processed = test_processed_with_context.iloc[context_size:]

    # Get feature and target columns
    target_cols = [col for col in train_processed.columns if col.startswith('target_t+')]
    feature_cols = [col for col in train_processed.columns if col not in target_cols 
                    and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # print(train_processed[feature_cols].head())
    # print(train_processed[feature_cols].tail())
    # print(test_processed[feature_cols].head())
    # print(test_processed[feature_cols].tail())

    train_df = train_processed.dropna()
    test_df = test_processed.dropna()

    return train_df, test_df, feature_cols, target_cols


def _create_datasets(train_df, test_df, feature_cols, target_cols):
    """Create training and test datasets."""
    config = get_config()
    window_size = config.get_window_size()
    
    train_ds = SlidingWindowDataset(train_df[feature_cols + target_cols], 
                                   training_window_size=window_size, 
                                   horizon_size=HORIZON, 
                                   feature_cols=feature_cols)
    test_ds = SlidingWindowDataset(test_df[feature_cols + target_cols], 
                                  training_window_size=window_size, 
                                  horizon_size=HORIZON, 
                                  feature_cols=feature_cols)
    
    return train_ds, test_ds


def _evaluate_model(model, test_loader):
    """Evaluate model and return validation loss."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = criterion(model(batch_x), batch_y)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def train_with_grid_search(verbose: bool=True):
    """
    Train models with different hyperparameter combinations from a grid.
    
    Args:
        verbose: Whether to print training progress
    
    Returns:
        best_model: The model with the best validation performance
        best_params: The hyperparameters that produced the best model
        results: Dictionary with training results for all parameter combinations
    """
    # Get configuration and hyperparameters
    grid_params = HYPERPARAMETER_GRIDS['Transformer']
    
    # Create parameter combinations
    keys = grid_params.keys()
    values = grid_params.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Prepare data
    train_df, test_df, feature_cols, target_cols = _prepare_training_data()
    train_ds, test_ds = _create_datasets(train_df, test_df, feature_cols, target_cols)

    # Initialize tracking
    results = []
    best_model = None
    best_score = float('inf')
    best_params = None

    total_combinations = len(combinations)
    start_time = time.time()

    # Grid search loop
    for i, params in enumerate(combinations):
        current_start = time.time()
        # --- prunning suboptimal ---
        if best_score != float('inf') and len(results) > 0:
            pass

        # --- time estimation ---
        if i > 0:
            avg_time_per_run = (time.time() - start_time) / i
            remaining_runs = total_combinations - i
            est_remaining = avg_time_per_run * remaining_runs
            readable_time = str(timedelta(seconds=int(est_remaining)))
            logger.info(f'Progress: {i}/{total_combinations} | Est. time remaining: {readable_time}')
        else:
            logger.info(f"Progress: 0/{total_combinations} | First run in progress...")

        # Get parameters with defaults
        lr = params.get('learning_rate', TRAINING_PARAMS['learning_rate'])
        epochs = params.get('epochs', TRAINING_PARAMS['epochs'])
        batch_size = params.get('batch_size', BATCH_SIZE)
        
        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Initialize model with hyperparameters
        model = BitcoinTransformer(
            input_dim=len(feature_cols),
            model_dim=params.get('model_dim', 64),
            n_heads=params.get('n_heads', 4),
            n_layers=params.get('n_layers', 2),
            dropout=params.get('dropout', 0.1)
        ).to(device)

        if verbose:
            logger.info(f"Testing combination: {params}")

        # Train model
        train_model(model, train_loader, epochs=epochs, lr=lr, verbose=verbose)

        # Evaluate model
        avg_loss = _evaluate_model(model, test_loader)

        # --- Dynamic Stopping ---
        if np.isnan(avg_loss) or avg_loss > (best_score * 10 if best_score != float('inf') else 100):
            logger.warning(f"Skipping extremely poor result: {avg_loss:.6f}")
            continue

        params['validation_loss'] = avg_loss
        results.append(params)

        if verbose:
            logger.info(f"Validation Loss: {avg_loss:.6f}")

        # Track best model
        if avg_loss < best_score:
            best_score = avg_loss
            best_params = params
            best_model = model
            save_best_params(best_params)

    # Save best model and parameters
    if best_model is not None:
        save_model_and_pipeline(best_model, transformer_pipeline)

    total_duration = str(timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'Grid search complete in {total_duration}. Best loss: {best_score:.6f}')

    return best_model, best_params, results


def run_transformer_inference(model_path: str=None, pipeline_path: str=None):
    """Main function to run production inference only (no training)."""
    
    logger.debug(f"Running transformer inference with model_path={model_path}, pipeline_path={pipeline_path}")
    
    # Get dynamic configuration
    config = get_config()
    column_to_predict = config.get_column_to_predict()
    
    # Prepare data for inference
    train_df, test_df, feature_cols, target_cols = _prepare_training_data()
    train_ds, _ = _create_datasets(train_df, test_df, feature_cols, target_cols)

    best_params = load_best_params(f'models/transformer/{column_to_predict}/best_params.json')

    # --- 3. MODEL INITIALIZATION ---
    if best_params:
        model = BitcoinTransformer(
            input_dim=len(feature_cols),
            model_dim=best_params.get('model_dim', 64),
            n_heads=best_params.get('n_heads', 4),
            n_layers=best_params.get('n_layers', 2),
            dropout=best_params.get('dropout', 0.1),
            horizon=HORIZON
        ).to(device)
    else:
        logger.warning("Using default model architecture (no best_params found)")
        model = BitcoinTransformer(input_dim=len(feature_cols), horizon=HORIZON).to(device)

    # Load trained model and pipeline
    model, pipeline = load_model_and_pipeline(model, model_path, pipeline_path)
    if pipeline is None:
        return None, None

    # --- 5. PREDICT ---
    pred_log_returns = predict_with_model(model, train_df, feature_cols)

    # --- 6. INVERSE TRANSFORM ---
    # Get the last HORIZON prices for inverse transformation based on COLUMN_TO_PREDICT
    if column_to_predict == 'Close_log_return':
        last_prices = train_df['Close'].tail(HORIZON)
    else:  # For volatility or other targets
        last_prices = train_df[column_to_predict.replace('_log_return', '')].tail(HORIZON)
    
    pred_prices = inverse_transform_predictions(pred_log_returns, last_prices)
    return pred_log_returns, pred_prices



