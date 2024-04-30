import json
from datetime import datetime
import os

def load_json_config(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def to_date(date_str: str) -> datetime.date:
    """Converts a string date to a datetime.date object."""
    return datetime.strptime(date_str, '%Y-%m-%d').date()

def validate_assets(assets: list) -> list:
    """Ensures that assets are in the correct format."""
    if not all(isinstance(asset, str) for asset in assets):
        raise ValueError("All assets must be strings")
    return assets

def validate_model_path(path: str) -> str:
    """Checks if the model file exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return path