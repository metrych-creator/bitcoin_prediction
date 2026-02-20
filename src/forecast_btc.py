import joblib
import requests
import pandas as pd
import numpy as np
import time
from .data_processor import transform_data
# Add the project root to Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def count_days_since_last_candle(df=None):
    """Counts days since last candle till today."""
    if df is None:
        df = pd.read_csv('data/Bitcoin_history_data.csv')
    last_date_raw = df['Date'].iloc[-1]
    last_date = pd.to_datetime(last_date_raw)
    current_date = pd.Timestamp.now()
    days_since_last_candle = (current_date - last_date).days
    return days_since_last_candle


def get_crypto_data(symbol="BTCUSDT", interval="1d", limit=500):
    """Downloads OHLCV data from Binance.US API for a given cryptocurrency symbol, time interval, and data limit."""
    base_url = "https://api.binance.us/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit+1
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print("Error during downloading data:", response.text)
        return None

    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_Time', 'Quote_Volume', 'Trades', 'Takers_Buy_Base', 'Takers_Buy_Quote', 'Ignore'
    ])

    # Choose only columns 0-5 (Date, O, H, L, C, V)
    df = pd.DataFrame(data).iloc[:, range(6)]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    df['Date'] = pd.to_datetime(df['Date'], unit='ms')

    return df


def run_production_inference(model_path: str, pipeline_path: str):
    # 1. Load model and pipeline
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    # 2. Get latest data
    days_to_get = count_days_since_last_candle()
    df_live = get_crypto_data(symbol="BTCUSDT", interval="1d", limit=days_to_get)
    if df_live is None or df_live.empty:
        print("Error: Failed to retrieve data from the API.")
        return None
    
    # 3. Transform data using the same pipeline as training
    try:
        _, _, X_live, _, _  = transform_data(df_live, test=None, pipeline=pipeline, verbose=False)
    except Exception as e:
        print(f"Error during data transformation: {e}")
        return None

    # 5. Make prediction
    prediction = model.predict(X_live)

    # 6. Result
    current_date = df_live['Date'].iloc[-1]
    print("-" * 30)
    print(f"PREDICTION RAPORT FOR: BTCUSDT")
    print(f"Date of last candle: {current_date}")
    print(f"Predicted volatility for tomorrow: {prediction[0]:.4f}")
    print("-" * 30)
    
    return prediction[0]

