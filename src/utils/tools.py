import os
import joblib
from sklearn.pipeline import Pipeline
import torch
from src.utils.logger_config import logger
from src.config_manager import get_config
import json

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
config = get_config()
column_to_predict = config.get_column_to_predict()

def save_model_and_pipeline(model, pipeline: Pipeline, model_path: str = None, pipeline_path: str = None):
    if model_path is None:
        model_path = f'models/transformer/{column_to_predict}/model.pkl'

    if pipeline_path is None:
        pipeline_path = f'models/transformer/{column_to_predict}/feature_pipeline.pkl'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Model and pipeline saved to {model_path} and {pipeline_path}")


def load_model_and_pipeline(model, model_path: str, pipeline_path: str):
    if model_path is None:
        model_path = f"models/transformer/{column_to_predict}/model.pkl"

    if pipeline_path is None:
        pipeline_path = f"models/transformer/{column_to_predict}/feature_pipeline.pkl"

    if os.path.exists(model_path) and os.path.exists(pipeline_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Model and pipeline loaded from {model_path} and {pipeline_path}")
        return model, pipeline
    else:
        logger.error(f"Model or pipeline not exists in model path: {model_path}, pipeline path: {pipeline_path}")
        return None, None


def save_best_params(best_params, params_path: str=None) -> None:
    if params_path is None:
        params_path = f"models/transformer/{column_to_predict}/best_params.json"

    os.makedirs(os.path.dirname(params_path), exist_ok=True)

    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Best params save: {best_params}")


def load_best_params(path: str = None):
    if path is None:
        path = f"models/transformer/{column_to_predict}/best_params.json"

    if os.path.exists(path):
        with open(path, 'r') as f:
            best_params = json.load(f)
            logger.info(f"Successfully loaded best params from {path}")
            return best_params
    else:
        logger.error(f'Best params path not exists: {path}')
        return None