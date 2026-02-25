"""
Results display module for Transformer model.
Extracts result display logic from my_transformer.py to separate concerns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Set up project paths for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import COLUMN_TO_PREDICT, HORIZON, WINDOW
from src.utils.plots import plot_price_forecast


def show_results(full_df: pd.DataFrame, processed_df_raw: pd.DataFrame, pred_log_returns: list, pred_prices: list = None, show_plot: bool=True) -> None:
    """
    Display prediction results.
    Extracted from my_transformer.py to separate result display concerns.
    
    Args:
        full_df: Original DataFrame with raw OHLCV data
        processed_df_raw: Processed DataFrame with features
        pred_log_returns: List of predicted log returns
        pred_prices: List of predicted prices (optional, will calculate if not provided)
    """
    print("\n" + "="*30)
    print(f"BITCOIN {COLUMN_TO_PREDICT} FORECAST")
    print("="*30)
    print(f"WINDOW: {WINDOW} DAYS")

    if pred_prices is None:
        # Calculate prices if not provided (backward compatibility)
        if COLUMN_TO_PREDICT == 'Close_log_return':
            _show_price_prediction(full_df, pred_log_returns, show_plot)
        else:
            _show_volatility_prediction(processed_df_raw, pred_log_returns, show_plot)
    else:
        # Use pre-calculated prices
        if COLUMN_TO_PREDICT == 'Close_log_return':
            _show_price_prediction_with_prices(full_df, pred_log_returns, pred_prices, show_plot)
        else:
            _show_volatility_prediction_with_prices(processed_df_raw, pred_log_returns, pred_prices, show_plot)


def _show_price_prediction_with_prices(full_df: pd.DataFrame, pred_log_returns: list, pred_prices: list, show_plot: bool=True) -> None:
    """Show price prediction results using pre-calculated prices."""
    last_price = full_df['Close'].iloc[-1]

    print(f"Last actual price: {last_price:,.2f} USD, from: {full_df.index[-1].date()}")

    for i, price in enumerate(pred_prices, 1):
        print(f"Day {i}: {price:,.2f} USD (Log-Return: {pred_log_returns[i-1]:.4f})")

    if show_plot:
        plot_price_forecast(last_price, pred_prices)


def _show_volatility_prediction_with_prices(processed_df_raw: pd.DataFrame, pred_log_returns: list, pred_prices: list, show_plot: bool=True) -> None:
    """Show volatility prediction results using pre-calculated prices."""
    last_volatility = processed_df_raw['Volatility_7'].iloc[-1] 
    last_volatility_pc = last_volatility * 100
    
    print(f"Last actual volatility: {last_volatility_pc:,.2f} %, from: {processed_df_raw.index[-1].date()}")

    for i, volatility in enumerate(pred_prices, 1):
        print(f"Day {i}: {volatility:,.3f} %")

    if show_plot:
        plot_price_forecast(last_volatility_pc, pred_prices)


def _show_price_prediction(full_df: pd.DataFrame, pred_log_returns: list, show_plot: bool=True) -> None:
    """Show price prediction results."""
    last_price = full_df['Close'].iloc[-1]
    predicted_prices = []
    current_price = last_price

    print(f"Last actual price: {last_price:,.2f} USD, from: {full_df.index[-1].date()}")

    for log_ret in pred_log_returns:
        current_price = current_price * np.exp(log_ret)
        predicted_prices.append(current_price)

    for i, price in enumerate(predicted_prices, 1):
        print(f"Day {i}: {price:,.2f} USD (Log-Return: {pred_log_returns[i-1]:.4f})")

    if show_plot:
        plot_price_forecast(last_price, predicted_prices)


def _show_volatility_prediction(processed_df_raw: pd.DataFrame, pred_log_returns: list, show_plot: bool=True) -> None:
    """Show volatility prediction results."""
    last_volatility = processed_df_raw['Volatility_7'].iloc[-1] 
    predicted_volatilities = []
    current_volatility = last_volatility
    last_volatility_pc = last_volatility * 100
    
    print(f"Last actual volatility: {last_volatility_pc:,.2f} %, from: {processed_df_raw.index[-1].date()}")

    for log_ret in pred_log_returns:
        current_volatility = current_volatility * np.exp(log_ret)
        predicted_volatilities.append(current_volatility * 100)

    for i, volatility in enumerate(predicted_volatilities, 1):
        print(f"Day {i}: {volatility:,.3f} %")

    if show_plot:
        plot_price_forecast(last_volatility_pc, predicted_volatilities)