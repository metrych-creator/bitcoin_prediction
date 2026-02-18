import sys
import streamlit as st
from src.forecast_btc import run_production_inference
from view.styles import NAV_BAR_HTML
from pathlib import Path
# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="BitForecaster", page_icon="images/bitcoin-logo.png", layout="wide")
# navigation bar
st.markdown(NAV_BAR_HTML, unsafe_allow_html=True)


st.header("Market Analysis & Predictions")
prediction = run_production_inference('models/best_model.pkl', 'models/feature_pipeline.pkl')
st.text(f"Predicted volatility for tomorrow: {prediction:.4f}")