"""Centralised logging configuration."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(
    name: str = "scinanoai",
    level: str | int | None = None,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configure and return a project logger.

    Idempotent: calling it twice with the same name returns the same instance
    without duplicating handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    resolved_level = level or os.getenv("SCINANO_LOG_LEVEL", "INFO")
    logger.setLevel(resolved_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
