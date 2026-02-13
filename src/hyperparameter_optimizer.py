"""
Hyperparameter optimization module for model comparison.
Provides functions to optimize hyperparameters using GridSearchCV with time series cross-validation.
"""

import json
import os
import joblib
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from src.config import HYPERPARAMETER_GRIDS, GRID_SEARCH_SETTINGS, OPTIMIZATION_SETTINGS, MODEL_SETTINGS

# Suppress warnings during hyperparameter optimization
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def optimize_hyperparameters(model, model_name, X_train, y_train):
    """
    Optimize hyperparameters for a given model using GridSearchCV.
    
    Args:
        model: The model instance to optimize
        model_name: String name of the model for config lookup
        X_train: Training features
        y_train: Training target
    
    Returns:
        tuple: (optimized_model, best_params)
    """
    if model_name in HYPERPARAMETER_GRIDS:
        print(f"Optimizing hyperparameters for {model_name}...")
        
        param_grid = HYPERPARAMETER_GRIDS[model_name]
        
        # Apply model-specific settings if available
        if model_name in MODEL_SETTINGS:
            for param, value in MODEL_SETTINGS[model_name].items():
                if hasattr(model, param):
                    setattr(model, param, value)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            **GRID_SEARCH_SETTINGS
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        
        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best CV score: {best_score:.4f}")
        
        _save_hyperparameters(model_name, best_params)
        _save_fitted_model(model_name, grid_search.best_estimator_)
        
        return grid_search.best_estimator_, best_params
    else:
        print(f"No hyperparameter grid defined for {model_name}, using default parameters")
        return model, None


def _save_hyperparameters(model_name, best_params):
    if OPTIMIZATION_SETTINGS['save_hyperparameters']:
        if not os.path.exists('results'):
            os.makedirs('results')
        
        params_file = f"results/{model_name}_best_params.json"
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"Saved best hyperparameters to: {params_file}")


def _save_fitted_model(model_name, best_estimator):
    if OPTIMIZATION_SETTINGS['save_models']:
        if not os.path.exists('results'):
            os.makedirs('results')
        
        model_file = f"models/{model_name}_best_model.pkl"
        joblib.dump(best_estimator, model_file)
        
        print(f"Saved fitted model to: {model_file}")


def load_best_hyperparameters(model_name):
    params_file = f"results/{model_name}_best_params.json"
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            return json.load(f)
    return None


def load_best_model(model_name):
    model_file = f"results/{model_name}_best_model.pkl"
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None