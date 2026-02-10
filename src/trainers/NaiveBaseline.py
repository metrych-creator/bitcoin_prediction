import numpy as np
import pandas as pd

class NaiveBaseline:
    """
    Naive model to predict last value (t-1).
    It predicts today what was yesterday.
    """
    def __init__(self, feature_col='Close_log_return'):
        self.feature_col = feature_col
        self.last_train_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.last_value_ = X[self.feature_col].iloc[-1]
        return self
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # t+1
        preds = X[self.feature_col].shift(1)

        # fill first row with last train value
        if self.last_value_ is not None:
            preds.iloc[0] = self.last_value_

        return preds