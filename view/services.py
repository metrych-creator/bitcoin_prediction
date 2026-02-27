from src.config_manager import get_config
from src.transformer_main import run_transformer_inference


class PredictionService:
    def __init__(self):
        self.config = get_config()
    
    def validate_params(self, window, horizon):
        errors = []
        if window < 7 or window > 2*365:
            errors.append("Window must be between 7 and 730 days (2 years).")
        if horizon < 1 or horizon > 365:
            errors.append("Horizon must be between 1 and 365 days.")
            
        if horizon > window:
            errors.append("Horizon cannot be longer than the training window.")
        
        return errors
    
    def get_predictions(self):
        return run_transformer_inference()
    
    def calculate_portfolio_outcome(self, amount, current_price, predicted_price):
        return (predicted_price - current_price) * amount
    

