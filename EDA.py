from plots import plot_close_price_by_time, plot_decomposition, plot_acf, plot_pacf
import pandas as pd
from src.data_loader import prepare_data, transform_data


def make_EDA(df):
    train, test = prepare_data(df)

    # Close
    plot_close_price_by_time(pd.concat([train, test]), show=False)

    train_transformed, test_transformed = transform_data(train, test)

    # Log Returns
    plot_close_price_by_time(pd.concat([train_transformed, test_transformed]), title='BTC Log Returns', pic_name='log_returns', show=False)

    plot_acf(train['Close'], max_lags=50, pic_name='ACF_Raw_Data', title='ACF - Raw Data')
    plot_acf(train_transformed['Close'], max_lags=50, pic_name="ACF_stationary", title="ACF - Stationanary Data")

    plot_pacf(train_transformed['Close'], n_lags=20, pic_name='PACF', title='Partial Autocorrelation Functions (PACF)')

    plot_decomposition(train_transformed, period=7)
    plot_decomposition(train_transformed, period=365)


df = pd.read_csv('data/Bitcoin_history_data.csv')
make_EDA(df)