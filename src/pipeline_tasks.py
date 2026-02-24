from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from torch import nn
from src.config import HORIZON, WINDOW
from src.utils.tools import add_bollinger_bands_prc, add_rsi
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


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
    def __init__(self, target_col='Close_log_return', horizon: int=1):
        self.target_col = target_col
        self.horizon = horizon

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        for i in range(1, self.horizon + 1):
            X[f'target_t+{i}'] = X[self.target_col].shift(-i)
        return X
    

class SlidingWindowDataset(Dataset):
    def __init__(self, data, training_window_size: int=WINDOW, horizon_size: int=HORIZON, feature_cols: list=None):
        self.scaler = MinMaxScaler()

        self.training_window_size = training_window_size
        self.horizon_size = horizon_size

        # target columns (e.g. target_t+1, target_t+2...)
        target_cols = [f'target_t+{i}' for i in range(1, self.horizon_size + 1)]

        # feature columns (exclude target and raw price columns)
        if feature_cols:
            X_raw = data[feature_cols].values
        else:
            X_raw = data.drop(columns=target_cols).values
            
        # split target columns and features
        y_raw = data[target_cols].values
        
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.X = self.scaler_x.fit_transform(X_raw)
        self.y = self.scaler_y.fit_transform(y_raw)
        
        # Add input_dim attribute for compatibility with refactored code
        self.input_dim = X_raw.shape[1] if len(X_raw.shape) > 1 else 1

    def __len__(self):
        return len(self.X) - self.training_window_size

    def __getitem__(self, idx):
        # x - window size of days, y - next day
        x = self.X[idx : idx + self.training_window_size]
        y = self.y [idx + self.training_window_size - 1]
        return torch.FloatTensor(x), torch.FloatTensor(y)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
            Positional encoding for transformer models. Adds information about the position of each element in the sequence.
        Args:
            d_model: The dimension of the model (embedding size).
            max_len: The maximum sequence length (how many days/time steps back).
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return x
    

