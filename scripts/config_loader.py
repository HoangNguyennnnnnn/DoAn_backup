"""
Config Loader Utility

Loads and merges YAML configuration files with environment variable expansion.
Provides convenient access to training, hardware, and data configs.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and merge YAML configs with environment variable expansion."""
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """
        Load YAML config file with environment variable expansion.
        
        Supports ${ENV_VAR} syntax in config values.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            Parsed config dictionary
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        
        # Recursively expand environment variables
        ConfigLoader._expand_env_vars(config)
        return config
    
    @staticmethod
    def _expand_env_vars(config: Dict[str, Any]) -> None:
        """Recursively expand ${VAR} patterns in config values."""
        for key, value in config.items():
            if isinstance(value, dict):
                ConfigLoader._expand_env_vars(value)
            elif isinstance(value, str) and "${" in value:
                config[key] = os.path.expandvars(value)
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple config dictionaries (later overrides earlier)."""
        result = {}
        for config in configs:
            result.update(config)
        return result


# Example usage (stub for documentation)
if __name__ == "__main__":
    # Load hardware config
    hw_config = ConfigLoader.load_yaml("configs/hardware_p100.yaml")
    print("Hardware Config:", hw_config.get("hardware", {}).get("profile_name"))
    
    # Load training config
    train_config = ConfigLoader.load_yaml("configs/train_stage1.yaml")
    print("Model:", train_config.get("model", {}).get("name"))
    
    # Verify environment variable expansion
    print(f"Expanded OUTPUT_ROOT: {train_config.get('logging', {}).get('log_dir')}")
