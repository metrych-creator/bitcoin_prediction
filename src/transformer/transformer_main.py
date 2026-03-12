from itertools import product
import json
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import time
from datetime import timedelta

# Set up project paths for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.transformer.transformer_pipeline import transformer_pipeline
from src.utils.logger_config import logger
from src.utils.tools import load_best_params, load_model_and_pipeline, save_best_params, save_model_and_pipeline
from src.transformer.transformer_architecture import BitcoinTransformer
from src.data_processor import create_datasets, get_last_data, inverse_transform_predictions, load_full_dataset, prepare_predict_data, prepare_train_test_data
from src.config import HORIZON, HYPERPARAMETER_GRIDS, WINDOW, BATCH_SIZE, TRAINING_PARAMS
from src.config_manager import get_config
from src.pipeline_tasks import SlidingWindowDataset

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


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
    train_df, test_df, feature_cols, target_cols, trained_pipeline = prepare_train_test_data()
    train_ds, test_ds = create_datasets(train_df, test_df, feature_cols, target_cols)

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
        model.run_training_loop(train_loader, epochs=epochs, lr=lr, verbose=verbose)

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
        save_model_and_pipeline(best_model, trained_pipeline)

    total_duration = str(timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'Grid search complete in {total_duration}. Best loss: {best_score:.6f}')

    return best_model, best_params, results


def run_transformer_inference(model_path: str=None, pipeline_path: str=None):
    """Main function to run production inference only (no training)."""
    
    logger.debug(f"Running transformer inference with model_path={model_path}, pipeline_path={pipeline_path}")
    
    # Get dynamic configuration
    config = get_config()
    column_to_predict = config.get_column_to_predict()
    best_params = load_best_params(f'models/transformer/{column_to_predict}/best_params.json')
    _, _, feature_cols, _, _ = prepare_train_test_data() # get feature_cols


    # --- 3. MODEL INITIALIZATION ---
    if best_params:
        model_shape = BitcoinTransformer(
            input_dim=len(feature_cols),
            model_dim=best_params.get('model_dim', 64),
            n_heads=best_params.get('n_heads', 4),
            n_layers=best_params.get('n_layers', 2),
            dropout=best_params.get('dropout', 0.1),
            horizon=HORIZON
        ).to(device)
    else:
        logger.warning("Using default model architecture (no best_params found)")
        model_shape = BitcoinTransformer(input_dim=len(feature_cols), horizon=HORIZON).to(device)


    # Load trained model and pipeline
    model, pipeline = load_model_and_pipeline(model_shape, model_path, pipeline_path)

    if pipeline is None:
        logger.info("Model/Pipeline not found. Training from scratch...")
        pipeline = model_shape.train_with_best_params()

    if pipeline is None: # if not found best_params
        return None, None

    # Prepare data for inference
    df_processed = prepare_predict_data(pipeline, window=config.get_window_size())

    # --- 5. PREDICT ---
    pred_log_returns = model.predict(df_processed)

    # --- 6. INVERSE TRANSFORM ---
    # Get the last HORIZON prices for inverse transformation based on COLUMN_TO_PREDICT
    df = load_full_dataset()
    if column_to_predict == 'Close_log_return':
        last_prices = df['Close'].tail(HORIZON)
    else:  # For volatility or other targets
        last_prices = df[column_to_predict.replace('_log_return', '')].tail(HORIZON)
    
    pred_prices = inverse_transform_predictions(pred_log_returns, last_prices)
    logger.info(f'Tomorrows prediction: {pred_prices[0]}')
    return pred_log_returns, pred_prices



