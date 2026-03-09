"""
Dynamic configuration manager for runtime parameter management.
Allows UI to override hardcoded configuration values.
"""

from dataclasses import dataclass
from typing import Any, Dict
from src.config import WINDOW, COLUMN_TO_PREDICT

@dataclass
class RuntimeConfig:
    """Runtime configuration that can be dynamically updated."""
    window_size: int = WINDOW
    column_to_predict: str = COLUMN_TO_PREDICT


class ConfigManager:
    """Manages configuration values that can be overridden at runtime."""
    
    def __init__(self):
        self._runtime_config = RuntimeConfig()
        self._defaults = {
            'window_size': WINDOW,
            'column_to_predict': COLUMN_TO_PREDICT
        }
    
    def update_from_ui(self, window_size: int, column_to_predict: str = None) -> None:
        """Update configuration from UI inputs."""
        self._runtime_config.window_size = window_size
        if column_to_predict:
            self._runtime_config.column_to_predict = column_to_predict
    
    def get_window_size(self) -> int:
        """Get current window size."""
        return self._runtime_config.window_size
    
    def get_column_to_predict(self) -> str:
        """Get current prediction target."""
        return self._runtime_config.column_to_predict
    
    def reset_to_defaults(self) -> None:
        """Reset all configuration to default values."""
        self._runtime_config = RuntimeConfig()
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all current configuration values."""
        return {
            'window_size': self.get_window_size(),
            'column_to_predict': self.get_column_to_predict()
        }


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager