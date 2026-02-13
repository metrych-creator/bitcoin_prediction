from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.base import BaseEstimator
from typing import Any
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class RegressionWrapper:
    """
    It standardizes the process of training and evaluating various regression estimators.
    This allows them to be compared in a single loop regardless of the library's origin.
    """
    def __init__(self, model: Any, name: str) -> None:
        self.model = model
        self.name = name
        self.metrics = {}

    def train(self, X_train: pd.DataFrame , y_train: pd.Series) -> None:
        print(f"Training of model: {self.name}...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return self.model.predict(X_test)
        
    def evaluate(self, y_test: pd.Series, predictions: pd.Series) -> dict:
        """Calculates MAE and MAPE errors on the test set and saves them in the self.metrics attribute."""

        self.metrics = {
            "model_name": self.name,
            "MAE": round(mean_absolute_error(y_test, predictions), 4),
            "MAPE": round(mean_absolute_percentage_error(y_test, predictions), 4),
        }
        return self.metrics
    

class ArimaWrapper(BaseEstimator):
    """
    Wrapper for ARIMA to make it compatible with sklearn-like API.
    """
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model_fit = None
        self.last_train_data = None

    def fit(self, X, y):
        # Ensure y has proper datetime index with frequency
        if hasattr(y.index, 'freq') and y.index.freq is None:
            # Try to infer frequency from the index
            try:
                y.index.freq = pd.infer_freq(y.index)
            except:
                # If inference fails, set a reasonable default frequency
                y.index.freq = 'D'  # Daily frequency
        
        self.model = ARIMA(y, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
        self.model_fit = self.model.fit()
        self.last_train_data = y
        return self

    def predict(self, X):
        forecast = self.model_fit.forecast(steps=len(X))
        return forecast
    
    
class ArimaxWrapper(BaseEstimator):
    def __init__(self, order=(2, 0, 2)):
        self.order = order
        self.model_fit = None

    def fit(self, X, y):
        # Ensure y has proper datetime index with frequency
        if hasattr(y.index, 'freq') and y.index.freq is None:
            # Try to infer frequency from the index
            try:
                y.index.freq = pd.infer_freq(y.index)
            except:
                # If inference fails, set a reasonable default frequency
                y.index.freq = 'D'  # Daily frequency
        
        self.model = ARIMA(y, exog=X, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
        self.model_fit = self.model.fit()
        return self

    def predict(self, X):
        return self.model_fit.get_forecast(steps=len(X), exog=X).predicted_mean
