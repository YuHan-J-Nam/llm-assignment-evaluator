"""
Logging utilities for the RAG system.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from rag_system.config import LOG_LEVEL, LOG_FILE


def setup_logger(name, log_file=LOG_FILE, level=LOG_LEVEL):
    """
    Set up a logger with console and file handlers.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (str): Logging level.
        
    Returns:
        logging.Logger: Configured logger object.
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger 