from src.data_loader import prepare_data, transform_data, inverse_transform_predictions
from src.wrapper import RegressionWrapper
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from src.trainers.NaiveBaseline import NaiveBaseline
from plots import plot_prediction, plot_close_price_by_time


if __name__ == '__main__':

    df = pd.read_csv('data/Bitcoin_history_data.csv')
    train, test = prepare_data(df)
    plot_close_price_by_time(pd.concat([train, test]), show=False)

    train_transformed, test_transformed = transform_data(train, test)

    # Lag transform, X - today, y - tomorrow
    X_train = train_transformed[['Close']].iloc[:-1]
    y_train = train_transformed['Close'].shift(-1).dropna()

    X_test = test_transformed[['Close']].iloc[:-1]
    y_test = test_transformed['Close'].shift(-1).dropna()
    test_prices_actual = test['Close'].iloc[1:]

    # 3. Models initialization
    baseline_model = NaiveBaseline()
    rf_regressor = RandomForestRegressor()

    models = [
        RegressionWrapper(baseline_model, "Baseline (Naive)"),
        RegressionWrapper(rf_regressor, "Random Forest Regressor"),
    ]

    results = []

    # 4. Running models
    for m in models:
        m.train(X_train=X_train, y_train=y_train) # fitting
        raw_preds = m.predict(X_test)

        preds = inverse_transform_predictions(
            raw_preds, 
            test_prices_actual, 
            last_train_price=train['Close'].iloc[-1])

        plot_prediction(test.iloc[1:], preds, m.name, show=False)
        res = m.evaluate(test['Close'].iloc[1:], preds)
        results.append(res)


    # 5. Comparing results
    df_results = pd.DataFrame(results)
    print("\n------- Model Comparison -------")
    print(df_results.sort_values(by="mae", ascending=False))