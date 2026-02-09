import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.tools import DateFormatter, TimeSeriesImputer, LogTransformer, DiffTransformer
from typing import Tuple, cast
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess stock data for time series regression.
    Format date columns, impute missing daily values, and optionally apply standard scaling.
    Data is split into train and test sets without shuffling to preserve temporal order.
    
    Args:
        path: Path to the input CSV file

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


def transform_data(train: pd.DataFrame, test: pd.DataFrame):
    pipeline = Pipeline([ 
        ('log', LogTransformer(column='Close')),
        ('diff', DiffTransformer(degree=1, verbose=True))
    ])
    
    train_transformed = pipeline.fit_transform(train)
    test_transformed = pipeline.transform(test)

    return cast(pd.DataFrame, train_transformed), cast (pd.DataFrame, test_transformed)


def inverse_transform_predictions(preds_log_diff: pd.Series, original_prices: pd.Series, last_train_price: float) -> pd.Series:
    """
    Converts forecasts from log-diff format back to dollars (USD)
    """
    prev_prices = original_prices.shift(1).copy()
    prev_prices.iloc[0] = last_train_price
    
    # log_price_pred = log(price_prev) + log_diff_pred
    log_preds = np.log(prev_prices.values.flatten()) + np.array(preds_log_diff).flatten()
    
    # return np.exp(log_preds)
    return pd.Series(np.exp(log_preds), index=original_prices.index)