import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.tools import DateFormatter, TimeSeriesImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, cast


def prepare_data(path: str, use_scaler: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess stock data for time series regression.
    Format date columns, impute missing daily values, and optionally apply standard scaling.
    Data is split into train and test sets without shuffling to preserve temporal order.
    
    Args:
        path: Path to the input CSV file
        use_scaler: Wheather to apply StandardScaler in the pipeline.

    Returns:
        A tuple containing the cleaned training and test DataFrames.
    """

    df = pd.read_csv(path)

    # Split data without shuffling for chronological order of time series
    train, test = train_test_split(
        df, test_size=0.3, random_state=42, shuffle=False)

    # Build pipeline steps
    pipeline = Pipeline([
        ('date_conv', DateFormatter(column_name='Date')),
        ('imputer', TimeSeriesImputer(freq='D')),
        ('scaler', StandardScaler() if use_scaler else None)
    ])

    # pipeline.set_output(transform='pandas')

    train_cleaned = pipeline.fit_transform(train)
    test_cleaned = pipeline.transform(test)

    return cast(pd.DataFrame, train_cleaned), cast(pd.DataFrame, test_cleaned)