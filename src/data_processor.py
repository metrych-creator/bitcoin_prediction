import pandas as pd
from sklearn.pipeline import Pipeline
from .pipeline_tasks import DateFormatter, FeatureEngineer, TechnicalFeaturesAdder, TimeSeriesImputer, LogTransformer, DiffTransformer, TimeSeriesShifter
from typing import Tuple, cast
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import COLUMN_TO_PREDICT

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess stock data for time series regression.
    Format date columns, impute missing daily values, and optionally apply standard scaling.
    Data is split into train and test sets without shuffling to preserve temporal order.
    
    Args:
        df: DataFrame with columns: 'Date', 'Close', 'High', 'Low', 'Open', 'Volume'

    Returns:
        A tuple containing the cleaned training and test DataFrames.
    """

    train, test = train_test_split(
        df, test_size=0.3, random_state=42, shuffle=False)

    # Build pipeline steps
    pipeline = Pipeline([
        ('date_conv', DateFormatter(column_name='Date')),
        ('imputer', TimeSeriesImputer(freq='D')),
    ])

    train_cleaned = pipeline.fit_transform(train)
    test_cleaned = pipeline.transform(test)

    return cast(pd.DataFrame, train_cleaned), cast(pd.DataFrame, test_cleaned)


def transform_data(train: pd.DataFrame, test: pd.DataFrame=None, pipeline: Pipeline=None, verbose: bool=True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Pipeline]:
    """
    Preprocess features for time series regression.
    Applies technical feature engineering to create a target variable for next-day prediction: 
        - log transformation, 
        - differencing,
        - time series shifting

    Supports:
    1. Training: Pass (train, test). Returns (X_train, y_train, X_test, y_test, pipeline).
    2. Live Inference: Pass (train=live_data, pipeline=trained_pipe). Returns (X_live).

    Returned columns:
    - RSI
    - BB_Percent
    - Open_log_return
    - High_log_return	
    - Low_log_return	
    - Close_log_return
    - Volume_log_return
    - MA_7
    - MA_30
    - MA_365
    - Volatility_7
    - Lag6
    """

    def extract_X_y(df):
        """Helper to drop non-features and select numeric types."""
        cols_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target_next_day']
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).select_dtypes(include=[np.number])
        y = df['target_next_day'] if 'target_next_day' in df.columns else None
        return X, y


    # --- LIVE INFERENCE MODE ---
    if pipeline is not None:
        # Use existing pipeline (already fitted)
        live_full = pipeline.transform(train)
        X_live, _ = extract_X_y(live_full.tail(1))
        return _, _, X_live, _, _


    # --- TRAINING MODE ---
    pipeline = Pipeline([
        ('tech_features', TechnicalFeaturesAdder()), # RSI, BB_Percent
        ('log', LogTransformer()),
        ('diff', DiffTransformer(degree=1, verbose=verbose)),
        ('engineer', FeatureEngineer()), # MA_7, MA_30, MA_365, Volatility_7
        ('shifter', TimeSeriesShifter(target_col=COLUMN_TO_PREDICT, shift=1, new_col_name='target_next_day'))
    ])
    
    # 1. Fit and transform training data
    train_full = pipeline.fit_transform(train)
    X_train, y_train = extract_X_y(train_full.dropna())
    
    # 2. Training Mode
    if test is not None:
        context_size = 365
        test_with_context = pd.concat([train.tail(context_size), test])
        test_full = pipeline.transform(test_with_context)
        X_test, y_test = extract_X_y(test_full.loc[test.index].dropna())

        return X_train, y_train, X_test, y_test, pipeline


def inverse_transform_predictions(preds_log_diff: pd.Series, original_prices: pd.Series) -> pd.Series:
    """
    Converts forecasts from log-diff format back to dollars (USD).
    """

    if COLUMN_TO_PREDICT not in ['Close_log_return']:
        return preds_log_diff
    
    preds = np.array(preds_log_diff).flatten()
    prices = np.array(original_prices).flatten()

    return np.exp(np.log(prices) + preds)