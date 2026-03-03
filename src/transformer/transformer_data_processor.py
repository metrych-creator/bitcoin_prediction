"""
Data processing module for Transformer model.
"""

from src.utils.logger_config import logger
import pandas as pd
from pathlib import Path
import sys

# Set up project paths for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor import count_days_since_last_candle, get_crypto_data_yahoo
from src.transformer.transformer_pipeline import transformer_pipeline


def prepare_joined_data() -> tuple:
    """
    Joins historical data and live data, applies transformations.
    
    Returns:
        tuple: (full_df, processed_df)
            - full_df: OHCLV DataFrame with historical and live data
            - processed_df: Transformed DataFrame with engineered features, log returns, 
                           technical indicators, and target variables
    """
    logger.info("Preparing joined dataset...")
    
    # Load historical data
    df = pd.read_csv('data/Bitcoin_history_data.csv')
    
    # Fetch the most recent candles from Yahoo Finance
    days_to_get = count_days_since_last_candle(df)
    df_live = get_crypto_data_yahoo(interval="1d", limit=days_to_get)
    
    # Merge datasets
    if df_live is not None and not df_live.empty:
        full_df = pd.concat([df, df_live], ignore_index=True)
    else:
        full_df = df
        
    # Clean and prepare data
    full_df['Date'] = pd.to_datetime(full_df['Date']).dt.normalize()
    full_df = full_df.drop_duplicates(subset=['Date'])
    full_df = full_df.sort_values('Date').set_index('Date')

    return full_df
