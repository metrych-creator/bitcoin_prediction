from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from typing import Any
import pandas as pd


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

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Calculates MAE and MAPE errors on the test set and saves them in the self.metrics attribute."""

        predictions = self.model.predict(X_test)
        
        self.metrics = {
            "model_name": self.name,
            "mae": round(mean_absolute_error(y_test, predictions), 4),
            "mape": round(mean_absolute_percentage_error(y_test, predictions), 4),
        }
        return self.metrics