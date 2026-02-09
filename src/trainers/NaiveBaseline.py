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
        self.last_value_ = y.iloc[-1]
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = X.shift(1)

        # use last value from train to predict first value in test
        if self.last_value_ is not None:
            preds.iloc[0] = self.last_value_

        return preds
    
    