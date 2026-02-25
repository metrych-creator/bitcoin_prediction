# Add the project root to Python path
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.forecast_btc import run_production_inference
from view.styles import NAV_BAR_HTML
from src.config import COLUMN_TO_PREDICT


st.set_page_config(page_title="BitForecaster", page_icon="images/bitcoin-logo.png", layout="wide")
# navigation bar
st.markdown(NAV_BAR_HTML, unsafe_allow_html=True)

st.header("Market Analysis & Predictions")
# one day prediction by LightGBM
# prediction = run_production_inference(f'models/{COLUMN_TO_PREDICT}/best_model.pkl', f'models/{COLUMN_TO_PREDICT}/feature_pipeline.pkl')
# st.text(f"Predicted {COLUMN_TO_PREDICT} for tomorrow: {prediction:.2f} {'$' if COLUMN_TO_PREDICT == 'Close_log_return' else '%'}")


run_production_inference(if_train=False, model_path=f'models/transformer/{COLUMN_TO_PREDICT}/model.pkl', pipeline_path=f'models/transformer/{COLUMN_TO_PREDICT}/feature_pipeline.pkl', epochs=5, lr=0.0001)
