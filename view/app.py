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

    ui.render_header()
    ui.render_inputs()

    if errors := service.validate_params(config.get_window_size(), st.session_state['horizon']):
        for error in errors:
            st.warning(error)

    if preds_new := service.get_predictions():
        ui.render_plots(*preds_new)
        ui.render_portfolio(*preds_new)


if __name__ == "__main__":
    main()
