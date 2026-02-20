"""
Configuration file for hyperparameter grids and model settings.
This allows for easy modification and management of hyperparameters.
"""

import numpy as np

# Hyperparameter grids for each model
HYPERPARAMETER_GRIDS = {
    'Random Forest Regressor': {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [3, 6, 10],
        'min_samples_split': [2, 5, 10],
    },
    'Ridge Regression': {
        'alpha': [10.0, 100.0, 200.0, 500.0]
    },
    'Light GBM': {
        'n_estimators': [25, 50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.005, 0.01, 0.1, 0.2],
    },
    'XGBoost': {
        'n_estimators': [25, 50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.005, 0.01, 0.1, 0.2],
    },
    'ARIMA': {
        'order': [(1, 0, 0), (0, 0, 1), (1, 1, 1), (2, 1, 2), (3, 1, 3), (5, 1, 0), (2, 0, 2)]
    },
    'ARIMAX': {
        'order': [(1, 0, 1), (2, 0, 2), (3, 0, 3), (2, 1, 2), (1, 1, 1)]
    }
}

# Cross-validation settings
CV_SETTINGS = {
    'n_splits': 3,
    'test_size': 0.2,
    'shuffle': False  # Important for time series data
}

# Grid search settings
GRID_SEARCH_SETTINGS = {
    'scoring': 'neg_mean_absolute_error',
    'cv': CV_SETTINGS['n_splits'],
    'n_jobs': -1,  # Use all available cores
    'verbose': 1
}

# Optimization settings
OPTIMIZATION_SETTINGS = {
    'enable_hyperparameter_tuning': True,
    'save_models': True,
    'save_hyperparameters': True
}

# Model-specific settings
MODEL_SETTINGS = {
    'Random Forest Regressor': {
        'random_state': 42,
        'n_jobs': -1,
        'max_features': 'sqrt'

    },
    'Ridge Regression': {
        'random_state': 42
    },
    'Light GBM': {
        'random_state': 42,
        'verbosity': -1,
        'min_gain_to_split': 0.1,  # Require minimum gain for splits
        'num_leaves': 5,  # Further limit number of leaves
        'colsample_bytree': 0.7,  # Increase column subsampling
        'lambda_l1': 0.1,  # L1 regularization
        'lambda_l2': 0.1,  # L2 regularization
        'min_child_samples': 50,
        'subsample': 0.7
    },
    'XGBoost': {
        'random_state': 42,
        'verbosity': 0,
        'subsample': 0.7
    }
}

COLUMN_TO_PREDICT = 'Close_log_return'
# COLUMN_TO_PREDICT = 'Volatility_7'
