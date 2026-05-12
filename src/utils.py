"""
Utility functions and constants for the fraud detection project
"""

import os
from pathlib import Path
from typing import List, Dict, Any


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Create directories if they don't exist
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Data configuration
DATASET_PATH = PROJECT_ROOT / "credit_card_fraud_dataset.csv"

# Model artifacts
MODEL_PATH = ARTIFACTS_DIR / "fraud_rf_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
REFERENCE_SCORES_PATH = ARTIFACTS_DIR / "reference_scores.npy"

# Feature configuration
NUMERIC_FEATURES = ["Amount", "MerchantID", "hour", "day", "weekday"]
CATEGORICAL_FEATURES = ["TransactionType", "Location"]
TARGET_COLUMN = "IsFraud"
ID_COLUMN = "TransactionID"
DATE_COLUMN = "TransactionDate"

# Model configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Risk thresholds
HIGH_RISK_THRESHOLD = 98.5  # percentile
MEDIUM_RISK_THRESHOLD = 95   # percentile

# Locations for the app
LOCATIONS = [
    "New York", "Los Angeles", "Houston", "Dallas",
    "Phoenix", "Philadelphia", "San Antonio",
    "San Diego", "San Jose"
]

# Risk labels
RISK_LABELS = {
    "high": ("🔴 HIGH RISK", "red"),
    "medium": ("🟠 MEDIUM RISK", "orange"),
    "low": ("🟢 LOW RISK", "green")
}


def ensure_artifacts_directory():
    """Ensure artifacts directory exists"""
    ARTIFACTS_DIR.mkdir(exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory"""
    return PROJECT_ROOT


def validate_environment():
    """Validate that the environment has required dependencies and files"""
    # Check if dataset exists
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    # Check if artifacts directory exists
    if not ARTIFACTS_DIR.exists():
        print(f"Creating artifacts directory at {ARTIFACTS_DIR}")
        ARTIFACTS_DIR.mkdir(parents=True)

    return True


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a value as percentage string"""
    return f"{value:.{decimals}f}%"


def calculate_percentile_rank(value: float, reference_values: list) -> float:
    """Calculate percentile rank of a value against reference values"""
    import numpy as np
    return (np.array(reference_values) < value).mean() * 100