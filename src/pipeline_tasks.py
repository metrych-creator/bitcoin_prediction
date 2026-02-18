from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

from src.utils.tools import add_bollinger_bands_prc, add_rsi

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
    """Impute lacking data."""
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
    def __init__(self, columns: list = None):
        self.columns = columns if columns else ['Open', 'High', 'Low', 'Close', 'Volume']

    def fit(self, X, y=None):
        self.columns = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.columns:
            X_copy[f'{col}_log'] = np.log(X_copy[col])
        return X_copy

    def set_output(self, transform=None):
        return self

    
class DiffTransformer(BaseEstimator, TransformerMixin):
    """Counts diff till time series is stationary."""
    def __init__(self, degree: int = 1, verbose: bool = False):
        self.degree = degree
        self.verbose = verbose

    def fit(self, X, y=None):
        # only cols with _log surfix
        self.columns_to_diff = [col for col in X.columns if col.endswith('_log')]
        return self

    def _check_stationarity(self, series: pd.Series, col_name: str):
        res = adfuller(series.dropna()) # ADF test for stationarity
        p_val = res[1]
        status = "STATIONARY" if p_val <= 0.05 else "NON-STATIONARY"
        print(f"[ADF Test] {col_name} | p-value: {p_val:.4f} -> {status}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.columns_to_diff:
            # replace log with log_returns
            new_name = col.replace('_log', '_log_return')
            X_copy[new_name] = X_copy[col]
            
            for _ in range(self.degree):
                X_copy[new_name] = X_copy[new_name].diff()
            
            if self.verbose:
                self._check_stationarity(X_copy[new_name], new_name)
            
            # remove _log cols
            X_copy = X_copy.drop(columns=[col])
                
        return X_copy

    def set_output(self, transform=None):
        return self


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds columns with: 
    MA (Moving Average) - 7 and 30 days,
    Volatility - 7 days svd,
    Lag - 6 day lagged value.
    """
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        X = X.copy()

        # Moving Averages
        X['MA_7'] = X['Close_log_return'].rolling(7).mean()
        X['MA_30'] = X['Close_log_return'].rolling(30).mean()
        X['MA_365'] = X['Close_log_return'].rolling(365).mean()

        
        # Volatility
        X['Volatility_7'] = X['Close_log_return'].rolling(window=7).std()
        
        # Lags (PACF)
        X['Lag6'] = X['Close_log_return'].shift(6)
        
        return X
    

class TechnicalFeaturesAdder(BaseEstimator, TransformerMixin):
    """Adds columns with:
    RSI (Relative Strength Index),
    Bollinger Bands Percentage - place where price currently is in Bollinger Bands."""
    def __init__(self, rsi_window=14, bb_window=20):
        self.rsi_window = rsi_window
        self.bb_window = bb_window

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        
        X = add_rsi(X, window=self.rsi_window)
        X = add_bollinger_bands_prc(X, window=self.bb_window)
        
        return X
    

class TimeSeriesShifter(BaseEstimator, TransformerMixin):
    """Compute real price in t+1 (tomorrow)."""
    def __init__(self, target_col='Close_log_return', shift: int = 1, new_col_name='target_next_day'):
        self.target_col = target_col
        self.shift = -shift
        self.new_target_col = new_col_name

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X[self.new_target_col] = X[self.target_col].shift(self.shift)
        return X