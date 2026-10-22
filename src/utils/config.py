from copy import deepcopy
from pathlib import Path
import yaml

def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(base_path, experiment_path):
    return deep_merge(load_yaml(base_path), load_yaml(experiment_path))
