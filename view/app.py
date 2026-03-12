import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from view.services import PredictionService
from src.transformer.transformer_main import run_transformer_inference, train_with_grid_search
from view.components import BitcoinUI
import streamlit as st
from src.config_manager import get_config
from src.utils.logger_config import logger


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
    logger.info("Starting Bitcoin Prediction App")
    config = get_config()
    ui = BitcoinUI()
    service = PredictionService()

    left, right = st.columns(spec=[1.5, 3.5], gap="large")
    preds = service.get_predictions()
    with left:
        ui.render_header()
        ui.render_inputs()
        ui.render_portfolio(preds[1])


    if errors := service.validate_params(config.get_window_size(), st.session_state['horizon']):
        for error in errors:
            st.warning(error)
            logger.error(f"Validation error: {error}")

    with right:
        ui.render_plot_type_buttons()
        ui.render_plots(*preds)


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    main()
    # train_with_grid_search()
    # run_transformer_inference()