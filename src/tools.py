from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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