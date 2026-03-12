
import streamlit as st

from src.config_manager import get_config
from src.data_processor import get_current_price
from src.config import COLUMN_TO_PREDICT
from view.plots import plot_predicted_percentage_prices, plot_predicted_prices
from view.styles import NAV_BAR_HTML

SIGN = '$' if COLUMN_TO_PREDICT == 'Close_log_return' else '%'

def compute_outcome(predicted_price: float, current_price: float, amount: float) -> float:
    if predicted_price is None:
        return 0
    return (predicted_price - current_price) * amount


class BitcoinUI:
    def __init__(self):
        self.config = get_config()
        
    def render_header(self):
        # Apply custom CSS styling
        st.markdown("""
            <style>
            div.stButton > button {
                height: 4em;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.set_page_config(page_title="BitForecaster", page_icon="images/bitcoin-logo.png", layout="wide")
        st.markdown(NAV_BAR_HTML, unsafe_allow_html=True)
        st.header("Bitcoin Market Predictions")
    
    def render_inputs(self):
        col1, col2 = st.columns(spec=[1, 1])
        with col1:
            window = st.number_input("Window Size", min_value=7, max_value=365*2, value=30)
            self.config.update_from_ui(window)
        with col2:
            horizon = st.number_input("Horizon Size", min_value=1, max_value=365, value=7)
            st.session_state['horizon'] = horizon
        
    
    def render_plot_type_buttons(self):
        # Add plot type buttons
        _, btn1, btn2, _ = st.columns(spec=[0.5, 1, 1, 2])
        with btn1:
            if st.button("Close price", width='stretch'):
                st.session_state.current_plot = "close_price"
        with btn2:
            if st.button("Percentage", width='stretch'):
                st.session_state.current_plot = "percentage"
        
    
    def render_plots(self, preds_log_return, preds):
        plot_type = st.session_state.get('current_plot', 'close_price')
        
        if plot_type == "close_price":
            st.subheader("Price Forecast")
            fig = plot_predicted_prices(preds, self.config.get_window_size(), st.session_state['horizon'])
            st.plotly_chart(fig, width='content')
        else:
            st.subheader("Percentage Change Forecast")
            fig = plot_predicted_percentage_prices(preds_log_return, st.session_state['horizon'])
            st.plotly_chart(fig, width='content')
    
    def render_portfolio(self, preds):
        st.header("Your Bitcoin Portfolio")
        bit_col, _ = st.columns(spec=[1.2, 0.8])
        
        with bit_col:
            bitcoin_amount = st.number_input(
                "Insert amount of your Bitcoins", 
                min_value=0.00001, 
                value=0.00001, 
                format="%.5f", 
                step=0.00001
            )
            if bitcoin_amount < 0.00001:
                st.warning("⚠️ The minimum amount is 0.00001 BTC")
                bitcoin_amount = max(bitcoin_amount, 0.00001)

        current_price = get_current_price()
        horizon_size = st.session_state['horizon']
        outcome = compute_outcome(preds[horizon_size-1], current_price, bitcoin_amount)
        outcome_label = "gain" if outcome > 0 else "loss"

        st.text(f"Current Bitcoin price: {current_price:.2f} $")

        if bitcoin_amount > 0:
            color = "red" if outcome < 0 else "green"
            st.markdown(
                f"Your predicted {outcome_label} after {horizon_size} days: "
                f":{color}[{outcome:.2f} {SIGN}]")
