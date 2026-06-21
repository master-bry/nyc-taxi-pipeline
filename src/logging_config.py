"""
Structured logging configuration for NYC Taxi Pipeline.

Provides JSON and standard logging with environment-based levels.
Usage: from src.logging_config import get_logger
        logger = get_logger(__name__)
"""

import logging
import logging.config
import json
import sys
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    use_json: bool = False,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (optional)
        use_json: Use JSON formatter for structured logging

    Example:
        setup_logging(level="DEBUG", log_dir="logs", use_json=True)
    """
    log_dir_path = Path(log_dir) if log_dir else None
    if log_dir_path:
        log_dir_path.mkdir(parents=True, exist_ok=True)

    # Determine formatter
    formatter_class = JSONFormatter if use_json else logging.Formatter
    fmt_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = (
        formatter_class() if use_json else logging.Formatter(fmt_string)
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir_path:
        file_handler = logging.FileHandler(log_dir_path / "app.log")
        file_handler.setLevel(level)
        file_formatter = (
            formatter_class() if use_json else logging.Formatter(fmt_string)
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    root_logger.setLevel(level)


def get_logger(name: str, extra_data: Optional[dict] = None) -> logging.LoggerAdapter:
    """
    Get a logger instance with optional extra context.

    Args:
        name: Logger name (typically __name__)
        extra_data: Extra data to include in all log records from this logger

    Returns:
        logging.LoggerAdapter: Configured logger

    Example:
        logger = get_logger(__name__, extra_data={"service": "dashboard"})
        logger.info("Dashboard started")
    """
    logger = logging.getLogger(name)

    if extra_data:
        return logging.LoggerAdapter(logger, {"extra_data": extra_data})
    return logger  # type: ignore


# Default setup
setup_logging()
