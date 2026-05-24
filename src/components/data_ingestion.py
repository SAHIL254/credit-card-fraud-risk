"""
Data Ingestion Component
Handles loading and basic analysis of credit card fraud dataset
"""

import pandas as pd
from pathlib import Path

from src.logger import get_logger
from src.exception import DataIngestionError, DataValidationError
from src.utils import DATASET_PATH


class DataIngestion:
    """
    Component for loading and analyzing credit card fraud data
    """

    def __init__(self, data_path: str = None):
        """
        Initialize data ingestion component

        Args:
            data_path: Path to the CSV dataset file
        """
        self.logger = get_logger(__name__)
        self.data_path = Path(data_path) if data_path else DATASET_PATH
        self.logger.info(f"DataIngestion initialized with path: {self.data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load credit card fraud dataset from CSV

        Returns:
            DataFrame containing the dataset
        """
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            self.logger.info(f"✅ Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            self.logger.error(f"Dataset not found at {self.data_path}")
            raise DataIngestionError(f"Dataset not found at {self.data_path}") from e
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise DataIngestionError(f"Error loading data: {str(e)}") from e

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataset

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with dataset statistics
        """
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "fraud_distribution": df["IsFraud"].value_counts().to_dict(),
            "fraud_percentage": (df["IsFraud"].value_counts(normalize=True) * 100).to_dict()
        }
        return info

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataset has required columns and structure

        Args:
            df: Input DataFrame

        Returns:
            True if valid, raises exception if invalid
        """
        # MerchantID is optional (not required by notebook workflow)
        required_columns = [
            "TransactionID", "TransactionDate", "Amount",
            "TransactionType", "Location", "IsFraud"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df.empty:
            raise ValueError("Dataset is empty")

        if "IsFraud" not in df.columns or df["IsFraud"].nunique() < 2:
            raise ValueError("Dataset must contain fraud labels with both classes")

        return True