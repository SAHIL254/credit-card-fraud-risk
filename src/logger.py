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

    # Configure logging with explicit UTF-8 encoding to avoid Unicode errors
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler (UTF-8)
    file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Stream handler (console) - ensure UTF-8 where possible
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    # Avoid adding duplicate handlers if setup_logging called multiple times
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(log_file) for h in root.handlers):
        root.setLevel(log_level)
        root.addHandler(file_handler)
        root.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)