import logging
import logging.handlers
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
LOG_MAX_BYTES_ENV = "AGENT_LOG_MAX_BYTES"
LOG_BACKUP_COUNT_ENV = "AGENT_LOG_BACKUP_COUNT"

def setup_logging(
    level: Optional[str] = None,
    format_style: Optional[str] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    use_config: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the trading agent.
    
    Args:
        level: Log level (debug, info, warning, error, critical)
        format_style: Log format style (simple, verbose)
        log_file: Optional file path for logging
        verbose: Enable verbose logging format
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    
    Returns:
        Configured root logger
    """
    # Load configuration from YAML if requested
    if use_config:
        try:
            from .config import get_config
            config = get_config()
            
            # Use config values as defaults, but allow function parameters to override
            if level is None:
                level = config.logging.level.lower() if config.logging.level else "info"
            if format_style is None:
                format_style = config.logging.format.lower() if config.logging.format else "simple"
            if log_file is None:
                log_file = config.logging.file_path if config.logging.file_path else None
            if max_bytes is None:
                max_bytes = config.logging.max_bytes if config.logging.max_bytes > 0 else 10485760
            if backup_count is None:
                backup_count = config.logging.backup_count if config.logging.backup_count > 0 else 5
        except Exception as e:
            print(f"Warning: Failed to load config, using environment variables: {e}", file=sys.stderr)
            use_config = False
    
    # Fallback to environment variables if config not available
    if not use_config:
        if level is None:
            level = os.getenv(LOG_LEVEL_ENV, "info").lower()
        if format_style is None:
            format_style = os.getenv(LOG_FORMAT_ENV, "simple").lower()
        if log_file is None:
            log_file = os.getenv(LOG_FILE_ENV)
        if max_bytes is None:
            max_bytes_str = os.getenv(LOG_MAX_BYTES_ENV, "10485760")
            max_bytes = int(max_bytes_str) if max_bytes_str else 10485760  # 10MB default
        if backup_count is None:
            backup_count_str = os.getenv(LOG_BACKUP_COUNT_ENV, "5")
            backup_count = int(backup_count_str) if backup_count_str else 5  # 5 backups default
    
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
    
    # Console handler (if enabled in config)
    if use_config:
        try:
            from .config import get_config
            config = get_config()
            console_enabled = config.logging.console_enabled
        except:
            console_enabled = True  # Default to enabled if config fails
    else:
        console_enabled = True  # Default to enabled for backward compatibility
    
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation (if enabled in config)
    if use_config:
        try:
            from .config import get_config
            config = get_config()
            file_enabled = config.logging.file_enabled
        except:
            file_enabled = bool(log_file)  # Use log_file parameter if config fails
    else:
        file_enabled = bool(log_file)  # Use log_file parameter for backward compatibility
    
    if file_enabled and log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Use RotatingFileHandler for log rotation
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file} (max: {max_bytes//1024//1024}MB, backups: {backup_count})")
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
