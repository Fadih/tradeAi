import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Log levels mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
VERBOSE_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"

# Environment variable for log level
LOG_LEVEL_ENV = "AGENT_LOG_LEVEL"
LOG_FORMAT_ENV = "AGENT_LOG_FORMAT"
LOG_FILE_ENV = "AGENT_LOG_FILE"

def setup_logging(
    level: Optional[str] = None,
    format_style: Optional[str] = None,
    log_file: Optional[str] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Setup logging configuration for the trading agent.
    
    Args:
        level: Log level (debug, info, warning, error, critical)
        format_style: Log format style (simple, verbose)
        log_file: Optional file path for logging
        verbose: Enable verbose logging format
    
    Returns:
        Configured root logger
    """
    # Get log level from env or use default
    if level is None:
        level = os.getenv(LOG_LEVEL_ENV, "info").lower()
    
    # Get format style from env or use default
    if format_style is None:
        format_style = os.getenv(LOG_FORMAT_ENV, "simple").lower()
    
    # Get log file from env
    if log_file is None:
        log_file = os.getenv(LOG_FILE_ENV)
    
    # Validate log level
    if level not in LOG_LEVELS:
        print(f"Warning: Invalid log level '{level}', using 'info'", file=sys.stderr)
        level = "info"
    
    # Set log level
    log_level = LOG_LEVELS[level]
    
    # Choose format
    if verbose or format_style == "verbose":
        log_format = VERBOSE_FORMAT
    else:
        log_format = DEFAULT_FORMAT
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            root_logger.error(f"Failed to setup file logging to {log_file}: {e}")
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging initialized - Level: {level.upper()}, Format: {format_style}")
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()
