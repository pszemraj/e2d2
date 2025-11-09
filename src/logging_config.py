"""Structured logging configuration for the E2D2 project.

This module provides a centralized logging infrastructure with proper
formatting, levels, and handlers to replace scattered print statements.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.constants import LOG_DATE_FORMAT, LOG_FORMAT


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Set up a logger with consistent formatting.

    Args:
        name: Name of the logger (typically __name__ from calling module).
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to log file. If None, only console logging is used.
        console: Whether to enable console (stdout) logging.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Training started")
        >>> logger.debug("Batch size: %d", batch_size)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with default configuration.

    This is a convenience function for most common use cases.

    Args:
        name: Name of the logger (typically __name__ from calling module).

    Returns:
        Logger instance with default INFO level and console output.

    Example:
        >>> from src.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class LoggerContext:
    """Context manager for temporarily changing logger level.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LoggerContext(logger, logging.DEBUG):
        ...     logger.debug("This will be logged")
        >>> logger.debug("This will not be logged if default level is INFO")
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        """Initialize context manager.

        Args:
            logger: Logger instance to modify.
            level: Temporary logging level to set.
        """
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level

    def __enter__(self) -> logging.Logger:
        """Enter context and set new logging level."""
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Exit context and restore original logging level."""
        self.logger.setLevel(self.old_level)
