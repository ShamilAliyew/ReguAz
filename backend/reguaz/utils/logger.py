"""Project-wide logger utility.

This module provides a single, reusable way to obtain a configured
``logging.Logger`` instance for any pipeline or module in the project
(e.g. embedding, ingestion, retrieval, RAG). Callers only need to
supply a logger name and a log file name; this module takes care of
directory creation, formatting, encoding, and handler de-duplication.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from backend.reguaz.config import LOGS_PATH as LOGS_DIR

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LEVEL = logging.INFO


def _ensure_logs_dir(logs_dir: Path) -> None:
    """Ensures the logs directory exists on disk.

    Args:
        logs_dir: Directory in which log files should be created.

    Raises:
        OSError: If the directory cannot be created (e.g. due to
            insufficient permissions).
    """
    logs_dir.mkdir(parents=True, exist_ok=True)


def _build_formatter() -> logging.Formatter:
    """Builds the standard log formatter used across the project.

    Returns:
        A ``logging.Formatter`` configured with timestamps, logger
        name, and log level in the message layout.
    """
    return logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)


def _has_handler_for_file(logger: logging.Logger, file_path: Path) -> bool:
    """Checks whether the logger already has a file handler for a path.

    Args:
        logger: Logger instance to inspect.
        file_path: Target log file path to look for.

    Returns:
        True if a ``FileHandler`` already writes to ``file_path``.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename) == file_path:
                return True
    return False


def _has_console_handler(logger: logging.Logger) -> bool:
    """Checks whether the logger already has a console (stream) handler.

    Args:
        logger: Logger instance to inspect.

    Returns:
        True if a non-file ``StreamHandler`` is already attached.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            return True
    return False


def get_logger(
    name: str,
    log_file: str,
    level: int = DEFAULT_LEVEL,
    logs_dir: str | Path = LOGS_DIR,
) -> logging.Logger:
    """Returns a configured logger that logs to both console and file.

    The logger writes UTF-8 encoded, timestamped log entries to
    ``<logs_dir>/<log_file>`` as well as to the console (stdout). The
    logs directory is created automatically if it does not exist.
    Calling this function multiple times with the same ``name`` and
    ``log_file`` will not attach duplicate handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling
            module.
        log_file: Name of the log file to write to (e.g.
            ``"qdrant_ingestion.log"``). The file is created inside the
            project's logs directory.
        level: Logging level to apply to the logger and its handlers.
            Defaults to ``logging.INFO``.
        logs_dir: Directory in which log files are stored. Defaults to
            ``backend/reguaz/logs``.

    Returns:
        A configured ``logging.Logger`` instance ready for use.

    Raises:
        OSError: If the logs directory cannot be created.
    """
    # Reconfigure sys.stdout to support UTF-8 on Windows consoles (e.g. Git Bash)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logs_dir = Path(logs_dir)
    _ensure_logs_dir(logs_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = _build_formatter()
    file_path = (logs_dir / log_file).resolve()

    if not _has_handler_for_file(logger, file_path):
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not _has_console_handler(logger):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Alias for backward compatibility with scripts referencing setup_logger
setup_logger = get_logger