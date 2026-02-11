from src.data_processor import prepare_data, transform_data, inverse_transform_predictions
from src.wrapper import RegressionWrapper
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from src.trainers.NaiveBaseline import NaiveBaseline
from plots import plot_prediction


if __name__ == '__main__':

    df = pd.read_csv('data/Bitcoin_history_data.csv')
    train, test = prepare_data(df)
    
    # train_transformed, test_transformed = transform_data(train, test)
    train, test = prepare_data(df)
    X_train, y_train, X_test, y_test = transform_data(train, test, verbose=False)

    # real($) in t
    test_original_prices = test.loc[X_test.index, 'Close']

    # real($) in t+1 [TARGET]
    raw_actual_usd = inverse_transform_predictions(y_test, test_original_prices)
    actual_tomorrow_usd = pd.Series(raw_actual_usd, index=X_test.index)

    # 3. Models initialization
    baseline_model = NaiveBaseline()
    rf_regressor = RandomForestRegressor()

    models = [
        RegressionWrapper(baseline_model, "Baseline (Naive)"),
        RegressionWrapper(rf_regressor, "Random Forest Regressor"),
    ]

    results = []
    fetures_importances = []
    feature_names = X_train.columns


    # 4. Running models
    for m in models:
        m.train(X_train=X_train, y_train=y_train) # fitting
        raw_preds = m.predict(X_test)
        preds = inverse_transform_predictions(
            raw_preds, 
            test_original_prices)
        plot_prediction(actual_tomorrow_usd, preds, m.name, show=False)
        res = m.evaluate(actual_tomorrow_usd, preds)
        results.append(res)

        # feature importance
        if m.name != "Baseline (Naive)":
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                "Importance": m.model.feature_importances_
            })
            importance_df.to_csv(f"data/feature_importance/{m.name}.csv", index=False)


    # 5. Comparing results
    df_results = pd.DataFrame(results)
    print("\n------- Model Comparison -------")
    print(df_results.sort_values(by="MAE", ascending=False))