import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import datetime
from turtle import st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from src.data_processor import get_last_prices

def plot_predicted_prices(predicted_prices: list, window_size: int=30):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # historical
    prices_hist = get_last_prices(window_size)
    dates_hist = pd.date_range(end=today, periods=len(prices_hist)).tolist()

    # predictions
    dates_preds = pd.date_range(start=today+timedelta(days=1), periods=len(predicted_prices)).to_list()
    fig = go.Figure()

    # historical
    fig.add_trace(go.Scatter(
        x=dates_hist, 
        y=prices_hist,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#A0A0A0', width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d, %Y}<br>Price: $%{y:,.2f}<extra></extra>'
    ))

    # dashed connection
    fig.add_trace(go.Scatter(
            x=[dates_hist[-1], dates_preds[0]], 
            y=[prices_hist[-1], predicted_prices[0]],
            mode='lines',
            name='Transition',
            line=dict(color='#29B094', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # predictions
    fig.add_trace(go.Scatter(
            x=dates_preds, 
            y=list(predicted_prices),
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#29B094', width=3),
            marker=dict(size=6),
            hovertemplate='%{x|%b %d}<br>Predicted: $%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            tickformat='%d %b', 
            tickangle=-45       
        ),
        yaxis=dict(
            title="Predicted Price ($)",
            tickformat="$,.0f"
        ),
        hovermode="x unified",
        template="plotly_white"
    )

    return fig



def plot_predicted_percentage_prices(predicted_prices, horizon_size=7):
    start_date = datetime.now() + timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=horizon_size).tolist()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, 
        y=predicted_prices*100,  # Convert to percentage
        mode='lines+markers',
        line=dict(color='#29B094', width=3),
        marker=dict(size=8),
        hovertemplate='%{x|%b %d, %Y}<br>Percentage Change: %{y:,.2f}%<extra></extra>'
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            tickformat='%d %b', 
            dtick="D1",         
            tickangle=-45       
        ),
        yaxis=dict(
            title="Predicted Price Change (%)",
            tickformat=".2f%"
        ),
        hovermode="x unified",
        template="plotly_white"
    )

    return fig
