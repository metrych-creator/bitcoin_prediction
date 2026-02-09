import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd

def plot_close_price_by_time(df):
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x='Date', y='Close', color='black')
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.title("Close price by time")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Ensures labels aren't cut off
    plt.savefig('plots/close_price.png')
    plt.show()


def plot_prediction(test: pd.DataFrame, y_pred: pd.Series, model_name: str):
    plt.figure(figsize=(10, 6))

    ax = sns.lineplot(x=test.index, y=test['Close'], color='royalblue', alpha=1, label = 'Actual')
    sns.lineplot(x=test.index, y=y_pred, color='darkorange', alpha=1, linewidth=0.3, label='Predicted', linestyle='--')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m %Y'))

    plt.title(f"Prediction price by time - model: {model_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Ensures labels aren't cut off
    plt.legend()

    plt.savefig(f'plots/prediction_{model_name}.png')
    plt.show()