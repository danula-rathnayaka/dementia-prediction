import os

import yaml


def load_config(config_path='config/config.yaml'):
    """Loads and returns the configuration dictionary."""
    if not os.path.exists(config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(base_dir, '..'))
        config_path = os.path.join(repo_root, 'config', 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at expected path: {config_path}")

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
