import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.utils.plots import plot_close_price_by_time, plot_decomposition, plot_acf, plot_pacf, plot_feature_importance
from src.data_processor import prepare_data, transform_data


def make_EDA(df, show: bool = True):
    train, test = prepare_data(df)

    # Close
    plot_close_price_by_time(pd.concat([train, test]), show=show)

    X_train, _, X_test, _, _ = transform_data(train, test, verbose=show)

    # Log Returns
    plot_close_price_by_time(pd.concat([X_train, X_test]), y='Close_log_return', title='BTC Log Returns', pic_name='log_returns', show=show)

    plot_acf(train['Close'], max_lags=50, pic_name='ACF_Raw_Data', title='ACF - Raw Data', show=show)
    plot_acf(X_train['Close_log_return'], max_lags=50, pic_name="ACF_stationary", title="ACF - Stationanary Data", show=show)

    plot_pacf(X_train['Close_log_return'], n_lags=20, pic_name='PACF', title='Partial Autocorrelation Functions (PACF)', show=show)

    plot_decomposition(X_train, period=7, show=show)
    plot_decomposition(X_train, period=365, show=show)

    # feature importance
    plot_feature_importance(show=True)


df = pd.read_csv('data/Bitcoin_history_data.csv')
make_EDA(df, show=False)