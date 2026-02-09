from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

class DateFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        
    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.column_name] = pd.to_datetime(X[self.column_name])
        X = X.set_index(self.column_name)
        return X.sort_values(self.column_name)


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D'):
        self.freq = freq

    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.resample(self.freq).asfreq() # add lacking dates
        X = X.ffill()
        return X
    

class LogTransformer(BaseEstimator, TransformerMixin):
    """Applies log transformation to the specified column."""
    def __init__(self, column: str = 'Close'):
        self.column = column

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        X_copy[self.column] = np.log(X_copy[self.column])
        return X_copy

    def set_output(self, transform=None):
        return self
    

class DiffTransformer(BaseEstimator, TransformerMixin):
    """Performs differentiation and optionally checks stationarity."""
    def __init__(self, column: str = 'Close', degree: int = 1, verbose: bool = False):
        self.column = column
        self.degree = degree
        self.verbose = verbose

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def _check_stationarity(self, series: pd.Series):
        res = adfuller(series.dropna())
        p_val = res[1]
        status = "STATIONARY" if p_val <= 0.05 else "NON-STATIONARY"
        print(f"[ADF Test] Degree {self.degree} | p-value: {p_val:.4f} -> {status}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for _ in range(self.degree):
            X_copy[self.column] = X_copy[self.column].diff()
        
        if self.verbose:
            self._check_stationarity(X_copy[self.column])
            
        return X_copy.fillna(0)

    def set_output(self, transform=None):
        return self