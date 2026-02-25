import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from view.plots import plot_predicted_percentage_prices, plot_predicted_prices
from src.data_processor import get_current_price
import streamlit as st
from src.transformer_main import run_transformer_inference
from view.styles import NAV_BAR_HTML
from src.config import COLUMN_TO_PREDICT

SIGN = '$' if COLUMN_TO_PREDICT == 'Close_log_return' else '%'
st.markdown("""
    <style>
    div.stButton > button {
        height: 4em;
        width: 12em;       
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

def validate_params(window, horizon):
    errors = []
    if window < 7 or window > 365:
        errors.append("Window must be between 7 and 365 days.")
    if horizon < 1 or horizon > 30:
        errors.append("Horizon must be between 1 and 30 days.")
        
    if horizon > window:
        errors.append("Horizon cannot be longer than the training window.")
        
    return errors

def compute_outcome(predicted_price: float, current_price: float, amount: float) -> float:
    if predicted_price is None:
        return 0
    return (predicted_price - current_price) * amount


st.set_page_config(page_title="BitForecaster", page_icon="images/bitcoin-logo.png", layout="wide")
st.markdown(NAV_BAR_HTML, unsafe_allow_html=True)
st.header("Bitcoin Market Predictions")

# INPUTS
col1, col2, col3 = st.columns(spec=[1, 1, 4])
with col1:
    window_size = st.number_input("WINDOW SIZE: ", min_value=7, max_value=365, value=30, step=1, help="This is the number of past days that the model will use to make a prediction. \n For example, if you choose 30, the model will look at the last 30 days of data to predict the next value.")
    st.session_state['window_size'] = window_size
with col2:
    horizon_size = st.number_input("HORIZON SIZE: ", min_value=1, max_value=30, value=7, step=1, help="This is the number of days into the future that the model will predict.")
    st.session_state['horizon_size'] = horizon_size

validation_errors = validate_params(window_size, horizon_size)
if validation_errors:
    for err in validation_errors:
        st.warning(err)
else: # run moodel only if params are valid
    preds_log_return, preds = run_transformer_inference(if_train=False, epochs=5, lr=0.0001)
    st.text(f"Predicted close price for tomorrow: {preds[0]:.2f} {SIGN}")
    btn1, btn2, _ = st.columns(spec=[1, 1, 4], gap="small")
    if 'current_plot' not in st.session_state:
        st.session_state.current_plot = "close_price"
    with btn1:
        if st.button("Close price"):
            st.session_state.current_plot = "close_price"
    with btn2:
        if st.button("Percentage"):
            st.session_state.current_plot = "percentage"


if st.session_state.current_plot == "close_price":
    st.subheader("Price Forecast")
    fig = plot_predicted_prices(preds, window_size=st.session_state.horizon_size)
    st.plotly_chart(fig, use_container_width=False)

elif st.session_state.current_plot == "percentage":
    st.subheader("Percentage Change Forecast")
    fig = plot_predicted_percentage_prices(preds_log_return, horizon_size=st.session_state.horizon_size)
    st.plotly_chart(fig, use_container_width=False)


st.header("Your Bitcoin Portfolio")
bit_col, sep = st.columns(spec=[1, 5])
with bit_col:
    bitcoin_amount = st.number_input("Insert a amount of your Bitcoins", min_value=0.00001, value=0.00001, format="%.5f", step=0.00001)


current_price = get_current_price()
outcome = compute_outcome(preds[horizon_size-1], current_price, bitcoin_amount)
outcome_label = "gain" if outcome > 0 else "loss"


st.text(f"Current Bitcoin price: {current_price:.2f} $")

if bitcoin_amount > 0:
    color = "red" if outcome < 0 else "green"
    st.markdown(
        f"Your predicted {outcome_label} after {horizon_size} days: "
        f":{color}[{outcome:.2f} {SIGN}]")
