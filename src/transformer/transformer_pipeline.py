from sklearn.pipeline import Pipeline
from src.config import COLUMN_TO_PREDICT, HORIZON
from src.pipeline_tasks import DataFrameScaler, DiffTransformer, FeatureEngineer, LogTransformer, PositionalEncoding, TechnicalFeaturesAdder, TimeSeriesShifter, TimeSeriesImputer


transformer_pipeline = Pipeline([
        # ('date_formatter', DateFormatter(column_name='Date')),
        ('imputer', TimeSeriesImputer(freq='D')), # Fill missing dates
        ('tech_features', TechnicalFeaturesAdder()), # RSI, Bollinger Bands
        ('log_transformer', LogTransformer(columns=['Open', 'High', 'Low', 'Close', 'Volume'])),
        ('diff_transformer', DiffTransformer(degree=1, verbose=False)), # Make series stationary
        ('feature_engineer', FeatureEngineer()), # MAs, Volatility, Lags
        # ('data_scaler', DataFrameScaler()),
        ('shifter', TimeSeriesShifter(target_col=COLUMN_TO_PREDICT, horizon=HORIZON))
    ])