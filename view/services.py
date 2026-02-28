from src.config_manager import get_config
from src.transformer_main import run_transformer_inference
from src.utils.logger_config import logger


class PredictionService:
    """Service class for handling prediction-related operations."""
    
    def __init__(self):
        self.config = get_config()
        logger.info("PredictionService initialized")
    
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
        try:
            logger.info("Fetching predictions from transformer model")
            return run_transformer_inference()
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return None, None
    
    def calculate_portfolio_outcome(self, amount, current_price, predicted_price):
        """Calculate bitcoin outcome based on predictions."""
        if predicted_price is None or current_price is None:
            return 0
        return (predicted_price - current_price) * amount
    

