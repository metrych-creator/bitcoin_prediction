from pathlib import Path
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf as sm_plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from src.config import COLUMN_TO_PREDICT
from src.data_processor import prepare_data, transform_data
from src.forecast_btc import count_days_since_last_candle, get_crypto_data

def plot_close_price_by_time(df: pd.DataFrame, y='Close', title: str ="BTC Close Price Over Time", pic_name : str='close_price', show: bool=True):
    fig = plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x='Date', y=y, color='black')
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Ensures labels aren't cut off
    plt.savefig(f'plots/{pic_name}.png')
    if show:
        plt.show()
    plt.close(fig)


def plot_prediction_with_residuals(actual: pd.Series, predicted: pd.Series, model_name: str, show: bool=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # prediction comparison plot
    sns.lineplot(x=actual.index, y=actual, color='royalblue', alpha=0.7, label='Actual', ax=ax1)
    sns.lineplot(x=actual.index, y=predicted, color='darkorange', alpha=1, linewidth=1, 
                 label='Predicted', linestyle='--', ax=ax1)
    
    ax1.set_title(f"Price Prediction Comparison - model: {model_name}", fontsize=14)
    ax1.set_ylabel("Price [USD]")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # residuals plot
    residuals = actual - predicted
    ax2.plot(actual.index, residuals, color='crimson', label='Error (Actual - Pred)', alpha=0.8)
    ax2.fill_between(actual.index, residuals, 0, color='crimson', alpha=0.1)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.6)
    
    ax2.set_title("Prediction Errors (Residuals)", fontsize=12)
    ax2.set_ylabel("Error [USD]")
    ax2.grid(True, alpha=0.2)

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m %Y'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    folder_path = Path("plots/full_predictions/") / COLUMN_TO_PREDICT
    folder_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder_path / f'{model_name}.png', bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close(fig)


def plot_acf(df: pd.DataFrame, max_lags: int, pic_name: str, title: str = "Autocorrelation Plot", show: bool=True):
    fig = plt.figure(figsize=(10, 6))
    plt.acorr(df, maxlags = max_lags)
    plt.title(title) 
    plt.xlabel("Lags")
    plt.grid(True)
    plt.savefig(f"plots/{pic_name}.png")
    if show:
        plt.show() 
    plt.close(fig)


def plot_pacf(df: pd.DataFrame, n_lags: int, pic_name: str, title: str = "PACF", show: bool=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    sm_plot_pacf(df, lags=n_lags, color='royalblue', ax=ax)
    plt.title(title)
    plt.xlabel('Lags')
    plt.ylabel('PACF')
    plt.grid(True)
    plt.savefig(f'plots/{pic_name}.png')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_decomposition(df: pd.DataFrame, model: str='additive', period: int=365, show: bool=True):
    # additive in stationarized data
    result = seasonal_decompose(df['Close_log_return'], model=model, period=period)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    
    fig.suptitle(f"Model Decomposition - Period {period}", fontsize=16)
    
    result.observed.plot(ax=ax1, color='black', title='Observed (Original Data)')
    result.trend.plot(ax=ax2, color='blue', title='Trend (Long-term movement)')
    result.seasonal.plot(ax=ax3, color='green', title='Seasonality (Repeating patterns)', )
    result.resid.plot(ax=ax4, color='red', style='.', title='Residuals (Random Noise)')
    plt.tight_layout()
    plt.savefig(f'plots/model_decomposition_stationary_p_{period}.png')

    if show:
        plt.show()
    plt.close()


def plot_feature_importance(show: bool=True):
    folder_path = 'data/feature_importance/'

    for col in os.listdir(folder_path):
        my_path = os.path.join(folder_path, col)

        if len(os.listdir(my_path)) == 0:
            raise(FileNotFoundError)
        else:
            for file in os.listdir(my_path):
                name = file.split('.')[0]
                df = pd.read_csv(os.path.join(my_path, file), sep=',')
                df = df.sort_values(by='Importance', ascending=False)
                fig = plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x='Feature', y='Importance', color='royalblue')
                plt.title(f'Feature Importance of model: {name}', pad=20)
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.xticks(rotation=30)
                plt.tight_layout()
                plt.grid(axis='y', linestyle='--', color='lightblue')

                plot_path = Path("plots/feature_importance/") / col
                plot_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(f'{plot_path}/{name}.png', bbox_inches='tight')

                if show:
                    plt.show()
                plt.close(fig)


def plot_volatility_over_time(df: pd.DataFrame, show: bool=True):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Volatility_7'], color='royalblue', label='Volatility')
    plt.title('Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.savefig(f'plots/volatility.png')
    if show:
        plt.show()
    plt.close(fig)
