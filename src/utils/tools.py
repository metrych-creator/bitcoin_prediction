import os
import joblib
from sklearn import pipeline
import torch
from src.utils.logger_config import logger
from src.config_manager import get_config
import json

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
config = get_config()
column_to_predict = config.get_column_to_predict()

def save_model_and_pipeline(model, pipeline: pipeline, model_path: str = None, pipeline_path: str = None):
    if model_path is None:
        model_path = f'models/transformer/{column_to_predict}/model.pkl'

    if pipeline_path is None:
        pipeline_path = f'models/transformer/{column_to_predict}/feature_pipeline.pkl'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Model and pipeline saved to {model_path} and {pipeline_path}")


def save_best_params(best_params, params_path: str=None) -> None:
    if params_path is None:
        params_path = f"models/transformer/{column_to_predict}/best_params.json"

    os.makedirs(os.path.dirname(params_path), exist_ok=True)

    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Best params save: {best_params}")


def load_model_and_pipeline(model, model_path: str, pipeline_path: str):
    if model_path and pipeline_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        loaded_pipeline = joblib.load(pipeline_path)
        logger.info(f"Model loaded from {model_path}")
        return loaded_pipeline
    else:
        return None
