"""
Logging utilities for the fraud detection project
"""

import logging
from pathlib import Path


def setup_logging(log_file: str = "fraud_detection.log", log_level: int = logging.INFO):
    """
    Setup logging configuration for the project

    Args:
        log_file: Path to log file
        log_level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)