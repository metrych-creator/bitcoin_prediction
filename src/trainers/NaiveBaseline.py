import numpy as np
import pandas as pd

class NaiveBaseline:
    """
    Naive model to predict last value (t-1).
    It predicts today what was yesterday.
    """
    def __init__(self):
        self.last_train_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.last_train_value = y.iloc[-1]

    def predict(self, X: pd.DataFrame) -> pd.Series :
        if isinstance(X, pd.Series):
            X = X.to_frame()
            
        preds = X.shift(1)

        if self.last_train_value is not None:
            preds.iloc[0] = self.last_train_value

        # preds.iloc[0] = self.last_train_values
        return preds
    
    