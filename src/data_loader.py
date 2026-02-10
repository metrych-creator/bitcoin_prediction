import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.pipeline_tasks import DateFormatter, FeatureEngineer, TechnicalFeaturesAdder, TimeSeriesImputer, LogTransformer, DiffTransformer, TimeSeriesShifter
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


def transform_data(train: pd.DataFrame, test: pd.DataFrame, verbose: bool=True):
    pipeline = Pipeline([
        ('tech_features', TechnicalFeaturesAdder()), # RSI, Bollinger Prc
        ('log', LogTransformer()),
        ('diff', DiffTransformer(degree=1, verbose=verbose)),
        ('engineer', FeatureEngineer()), # MA, Lags, Vol
        ('shifter', TimeSeriesShifter(target_col='Close_log_return'))
    ])
    
    train_transformed = pipeline.fit_transform(train).dropna()
    test_transformed = pipeline.transform(test).dropna()

    cols_to_drop = ['Open', 'High', 'Close', 'Low', 'Volume', 'target_next_day']

    X_train = train_transformed.drop(columns=cols_to_drop)
    y_train = train_transformed['target_next_day']
    
    X_test = test_transformed.drop(columns=cols_to_drop)
    y_test = test_transformed['target_next_day']
    
    return X_train, y_train, X_test, y_test


def inverse_transform_predictions(preds_log_diff: pd.Series, original_prices: pd.Series) -> pd.Series:
    """
    Converts forecasts from log-diff format back to dollars (USD).
    """

    preds = np.array(preds_log_diff).flatten()
    prices = np.array(original_prices).flatten()

    return np.exp(np.log(prices) + preds)