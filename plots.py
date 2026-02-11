import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf as sm_plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import pandas as pd

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


def plot_prediction(test: pd.DataFrame, y_pred: pd.Series, model_name: str, show: bool=True):
    fig = plt.figure(figsize=(10, 6))

    ax = sns.lineplot(x=test.index, y=test, color='royalblue', alpha=1, label = 'Actual')
    sns.lineplot(x=test.index, y=y_pred, color='darkorange', alpha=1, linewidth=0.3, label='Predicted', linestyle='--')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m %Y'))

    plt.title(f"Prediction price by time - model: {model_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Ensures labels aren't cut off
    plt.legend()
    plt.savefig(f'plots/prediction_{model_name}.png')
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
    path = 'data/feature_importance/'
    if len(os.listdir(path)) == 0:
        raise(FileNotFoundError)
    else:
        for file in os.listdir(path):
            name = file.split('.')[0]
            df = pd.read_csv(os.path.join(path, file), sep=',')
            df = df.sort_values(by='Importance', ascending=False)
            _, ax = plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='Feature', y='Importance', color='royalblue', ax=ax)
            plt.title(f'Feature Importance of model: {name}', pad=20)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', color='lightblue')
            plt.savefig(f'plots/feature_importance_{name}.png', bbox_inches='tight')

        if show:
            plt.show()
        plt.close()