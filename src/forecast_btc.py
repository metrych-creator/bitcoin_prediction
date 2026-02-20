import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import COLUMN_TO_PREDICT
import joblib
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import time
from src.data_processor import transform_data


def count_days_since_last_candle(df=None):
    """Counts days since last candle till today."""
    if df is None:
        df = pd.read_csv('data/Bitcoin_history_data.csv')
    last_date_raw = df['Date'].iloc[-1]
    last_date = pd.to_datetime(last_date_raw)
    current_date = pd.Timestamp.now()
    days_since_last_candle = (current_date - last_date).days
    return days_since_last_candle


def get_crypto_data_yahoo(symbol="BTC-USD", interval="1d", limit=500):
    """Downloads OHLCV data from Yahoo Finance API for a given cryptocurrency symbol, time interval, and data limit."""
    limit += 1
    df = yf.download(symbol, period=f"{limit}d", interval=interval)
    df = df.reset_index()

    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def run_production_inference(model_path: str, pipeline_path: str):
    # 1. Load model and pipeline
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    # 2. Get latest data
    days_to_get = count_days_since_last_candle()
    df_live = get_crypto_data_yahoo(interval="1d", limit=days_to_get)
    if df_live is None or df_live.empty:
        print("Error: Failed to retrieve data from the API.")
        return None
        
    # 3. Transform data using the same pipeline as training
    try:
        _, _, X_live, _, _  = transform_data(df_live, test=None, pipeline=pipeline, verbose=False)
    except Exception as e:
        print(f"Error during data transformation: {e}")
        return None

    print (X_live.head())
    # 5. Make prediction
    prediction = model.predict(X_live)

    # 6. Result
    current_date = df_live['Date'].iloc[-1]
    print("-" * 30)
    print(f"PREDICTION RAPORT FOR: BTC-USD")
    print(f"Date of last candle: {current_date}")
    print(f"Predicted volatility for tomorrow: {prediction[0]:.4f}")
    print("-" * 30)
    
    return prediction[0]


run_production_inference(model_path=f'models/{COLUMN_TO_PREDICT}/best_model.pkl', pipeline_path=f'models/{COLUMN_TO_PREDICT}/feature_pipeline.pkl')