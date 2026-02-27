import sys
from pathlib import Path

from view.services import PredictionService
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from view.components import BitcoinUI
import streamlit as st
from src.config_manager import get_config


st.markdown("""
    <style>
    div.stButton > button {
        height: 4em;
        width: 12em;       
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    config = get_config()
    ui = BitcoinUI()
    service = PredictionService()

    left, right = st.columns(spec=[1.5, 3.5], gap="large")
    preds = service.get_predictions()
    with left:
        ui.render_header()
        ui.render_inputs()
        ui.render_portfolio(*preds)


    if errors := service.validate_params(config.get_window_size(), st.session_state['horizon']):
        for error in errors:
            st.warning(error)

    with right:
        ui.render_plot_type_buttons()
        ui.render_plots(*preds)


if __name__ == "__main__":
    main()
