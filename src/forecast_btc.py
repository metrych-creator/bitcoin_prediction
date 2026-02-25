import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import COLUMN_TO_PREDICT
import joblib
import requests
import pandas as pd
import numpy as np
import time
from src.data_processor import count_days_since_last_candle, get_crypto_data_yahoo, transform_data
import src.pipeline_tasks as pipeline_tasks
sys.modules['pipeline_tasks'] = pipeline_tasks


def run_production_inference(model_path: str, pipeline_path: str):
    # 1. Load model and pipeline
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    # 2. Get latest data
    df = pd.read_csv('data/Bitcoin_history_data.csv')
    days_to_get = count_days_since_last_candle(df)
    df_live = get_crypto_data_yahoo(interval="1d", limit=days_to_get)
    if df_live is None or df_live.empty:
        print("Error: Failed to retrieve data from the API.")
        return None
    
    # 3. Get train df to calculate longer windows features
    df_train = pd.read_csv('data/Bitcoin_history_data.csv')
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_live = pd.concat([df_train, df_live], ignore_index=True)
        
    # 4. Transform data using the same pipeline as training
    try:
        _, _, X_live, _, _  = transform_data(df_live, test=None, pipeline=pipeline, verbose=False)
    except Exception as e:
        print(f"Error during data transformation: {e}")
        return None

    print (X_live.head())
    # 5. Make prediction
    prediction_raw = model.predict(X_live)

    if COLUMN_TO_PREDICT in ['Close_log_return']:
        # Cena_dzisiaj = Cena_wczoraj * exp(log_return)
        last_actual_price = df_live['Close'].iloc[-1]
        prediction = last_actual_price * np.exp(prediction_raw)

    # 6. Result
    current_date = df_live['Date'].iloc[-1]
    print("-" * 30)
    print(f"PREDICTION RAPORT FOR: BTC-USD")
    print(f"Date of last candle: {current_date.date()}")
    print(f"Predicted {COLUMN_TO_PREDICT} for tomorrow: {prediction[0]:.2f}", "$" if COLUMN_TO_PREDICT == 'Close_log_return' else "%")
    print("-" * 30)
    
    return prediction[0]


# run_production_inference(f'models/{COLUMN_TO_PREDICT}/best_model.pkl', f'models/{COLUMN_TO_PREDICT}/feature_pipeline.pkl')

