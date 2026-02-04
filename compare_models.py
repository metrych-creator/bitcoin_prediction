from src.data_loader import prepare_data
from src.wrapper import RegressionWrapper
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from src.trainers.NaiveBaseline import NaiveBaseline


if __name__ == '__main__':
    # 1. Load
    train, test = prepare_data('data/Bitcoin_history_data.csv', use_scaler=False)

    # 2. Split
    X_train = train[['Close']]  # DataFrame
    y_train = train['Close']    # Series

    X_test = test[['Close']]
    y_test = test['Close']

    # 3. Models initialization
    baseline_model = NaiveBaseline()
    other_model = RandomForestRegressor()

    models = [
        RegressionWrapper(baseline_model, "Baseline (Naive)"),
        RegressionWrapper(other_model, "Random Forest Regressor"),
    ]

    results = []

    # 4. Running models
    for m in models:
        m.train(X_train=X_train, y_train=y_train) # fitting
        res = m.evaluate(X_test, y_test)
        results.append(res)


    # 5. Comparing results
    df_results = pd.DataFrame(results)
    print("\n------- Model Comparison -------")
    print(df_results.sort_values(by="mae", ascending=False))