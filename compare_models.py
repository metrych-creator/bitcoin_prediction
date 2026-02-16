import joblib
from sklearn.linear_model import Ridge
from src.data_processor import prepare_data, transform_data, inverse_transform_predictions
from src.wrapper import ArimaWrapper, RegressionWrapper, ArimaxWrapper
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from src.trainers.NaiveBaseline import NaiveBaseline
from plots import plot_prediction_with_residuals
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
from src.config import HYPERPARAMETER_GRIDS, GRID_SEARCH_SETTINGS, MODEL_SETTINGS, OPTIMIZATION_SETTINGS
from src.hyperparameter_optimizer import load_best_model, optimize_hyperparameters
import time


def save_feature_importance(model, model_name, feature_names):
    importance_df = None
    
    # LightGBM 
    if model_name == "Light GBM":
        try:
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                "Importance": importance_values
            })
        except Exception as e:
            print(f"Error getting LightGBM feature importance: {e}")
    
    # Tree-based models
    elif hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            "Importance": importance_values
        })
    
    # ARIMAX models
    elif hasattr(model, 'model_fit') and hasattr(model.model_fit, 'params'):
        params = model.model_fit.params
        # For ARIMAX, we need to filter out AR/MA parameters and keep only exogenous variables
        exog_params = {k: v for k, v in params.items() if k.startswith('x1') or k.startswith('x2')}
        if exog_params:
            importance_df = pd.DataFrame({
                'Feature': list(exog_params.keys()),
                "Importance": [abs(v) for v in exog_params.values()]
            })

    # Ridge Regression
    elif hasattr(model, 'coef_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(model.coef_)
        })


    if importance_df is not None and not importance_df.empty:
        importance_df.to_csv(f"data/feature_importance/{m.name}.csv", index=False)




if __name__ == '__main__':
    # 1. Load data
    df = pd.read_csv('data/Bitcoin_history_data.csv')

    # 2. Preprocessing and Feature Engineering
    train, test = prepare_data(df)
    X_train, y_train, X_test, y_test, trained_pipe = transform_data(train, test, verbose=False)

    test_original_prices = test.loc[X_test.index, 'Close'] # real($) in t

    raw_actual_usd = inverse_transform_predictions(y_test, test_original_prices) # real($) in t+1 [TARGET]
    actual_tomorrow_usd = pd.Series(raw_actual_usd, index=X_test.index) # real($) in t+1 [TARGET] with correct index for evaluation and plotting

    # 3. Models initialization
    baseline_model = NaiveBaseline()
    rf_regressor = RandomForestRegressor()
    ridge = Ridge()
    light_gbm = LGBMRegressor(verbose=-1)
    xgb = XGBRegressor()
    arima = ArimaWrapper()
    arimax = ArimaxWrapper()

    models = [
        RegressionWrapper(baseline_model, "Baseline (Naive)"),
        RegressionWrapper(rf_regressor, "Random Forest Regressor"),
        RegressionWrapper(ridge, "Ridge Regression"),
        RegressionWrapper(light_gbm, "Light GBM"),
        RegressionWrapper(xgb, "XGBoost"),
        RegressionWrapper(arima, "ARIMA"),
        RegressionWrapper(arimax, "ARIMAX"),
    ]

    results = []
    feature_names = X_train.columns


    # 4. Running models with hyperparameter optimization
    for m in models:
        print(f"\n{'='*50}")
        print(f"Training model: {m.name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        if (OPTIMIZATION_SETTINGS['enable_hyperparameter_tuning'] and m.name in HYPERPARAMETER_GRIDS):
            optimized_model, best_params = optimize_hyperparameters(
                m.model, m.name, X_train, y_train
            )

            # Extract the best estimator from GridSearchCV
            if hasattr(optimized_model, 'best_estimator_'):
                m.model = optimized_model.best_estimator_
            else:
                m.model = optimized_model
        else:
            print(f"No hyperparameter optimization for {m.name}, using default parameters")
            m.train(X_train=X_train, y_train=y_train)
        
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")
        
        # 5. Make predictions with the best model
        raw_preds = m.predict(X_test)
        preds = inverse_transform_predictions(
            raw_preds, 
            test_original_prices)

        plot_prediction_with_residuals(actual_tomorrow_usd, preds, m.name, show=False)

        res = m.evaluate(actual_tomorrow_usd, preds)
        res['Training Time'] = training_time
        results.append(res)

        # 6. Feature Importance
        save_feature_importance(m.model, m.name, feature_names)


    # 7. Comparing results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="MAE", ascending=True).set_index('model_name').rename_axis('Model Name')
    print("-"*18, " Model Comparison ", "-"*18)
    print(df_results)
    df_results.to_csv("results/model_comparison.csv", index=True)

    # 8. Save best model for production
    best_model_name = df_results.index[0]
    best_model = load_best_model(best_model_name)
    if best_model is not None:
        joblib.dump(best_model, f"models/best_model.pkl")
    joblib.dump(trained_pipe, 'models/feature_pipeline.pkl')