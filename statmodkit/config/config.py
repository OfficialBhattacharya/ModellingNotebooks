from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path

class ModelConfig:
    """A class to handle model configuration settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ModelConfig.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Dictionary of configuration settings
        """
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration settings.
        
        Returns:
            Dictionary of default configuration settings
        """
        return {
            'preprocessing': {
                'high_unique_threshold': 0.5,
                'missing_value_strategy': 'median',
                'categorical_encoding': 'integer',
                'text_distance_metric': 'levenshtein'
            },
            'feature_selection': {
                'n_features': 'auto',
                'n_jobs': -1,
                'cv_folds': 5,
                'scoring': 'accuracy',
                'optimization_trials': 100
            },
            'model': {
                'test_size': 0.2,
                'random_state': 42,
                'problem_type': 'classification'
            },
            'visualization': {
                'style': 'seaborn',
                'figsize': [12, 8],
                'color_palette': 'deep'
            }
        }
        
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration.
        
        Returns:
            Dictionary of preprocessing settings
        """
        return self.config['preprocessing']
        
    def get_feature_selection_config(self) -> Dict[str, Any]:
        """
        Get feature selection configuration.
        
        Returns:
            Dictionary of feature selection settings
        """
        return self.config['feature_selection']
        
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary of model settings
        """
        return self.config['model']
        
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Get visualization configuration.
        
        Returns:
            Dictionary of visualization settings
        """
        return self.config['visualization']
        
    def save_config(self, path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration settings.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def _update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> None:
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    _update_dict(d[k], v)
                else:
                    d[k] = v
                    
        _update_dict(self.config, updates)
        
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set_config_value(self, key: str, value: Any) -> None:
        """
        Set a specific configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value 