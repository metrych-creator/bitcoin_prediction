from plots import plot_close_price_by_time, plot_decomposition, plot_acf, plot_pacf
import pandas as pd
from src.data_loader import prepare_data, transform_data


def make_EDA(df):
    train, test = prepare_data(df)

    # Close
    plot_close_price_by_time(pd.concat([train, test]), show=False)

    X_train, _, X_test, _ = transform_data(train, test, verbose=False)

    # Log Returns
    plot_close_price_by_time(pd.concat([X_train, X_test]), y='Close_log_return', title='BTC Log Returns', pic_name='log_returns', show=True)

    plot_acf(train['Close'], max_lags=50, pic_name='ACF_Raw_Data', title='ACF - Raw Data', show=True)
    plot_acf(X_train['Close_log_return'], max_lags=50, pic_name="ACF_stationary", title="ACF - Stationanary Data", show=True)

    plot_pacf(X_train['Close_log_return'], n_lags=20, pic_name='PACF', title='Partial Autocorrelation Functions (PACF)', show=True)

    plot_decomposition(X_train, period=7, show=True)
    plot_decomposition(X_train, period=365, show=True)


df = pd.read_csv('data/Bitcoin_history_data.csv')
make_EDA(df)