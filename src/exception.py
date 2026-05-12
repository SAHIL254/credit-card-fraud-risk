"""
Custom Exceptions for Fraud Detection Project
"""

class FraudDetectionException(Exception):
    """Base exception class for fraud detection project"""
    pass


class DataIngestionError(FraudDetectionException):
    """Exception raised for data ingestion errors"""
    pass


class DataValidationError(FraudDetectionException):
    """Exception raised for data validation errors"""
    pass


class FeatureEngineeringError(FraudDetectionException):
    """Exception raised for feature engineering errors"""
    pass


class ModelTrainingError(FraudDetectionException):
    """Exception raised for model training errors"""
    pass


class ModelEvaluationError(FraudDetectionException):
    """Exception raised for model evaluation errors"""
    pass


class PredictionError(FraudDetectionException):
    """Exception raised for prediction errors"""
    pass


class ArtifactNotFoundError(FraudDetectionException):
    """Exception raised when required artifacts are not found"""
    pass