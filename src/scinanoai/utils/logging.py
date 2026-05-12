"""Centralised logging configuration."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Third-party loggers we want to surface during long startup steps.
_NOISY_LIBRARIES: tuple[str, ...] = (
    "scinanoai",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "huggingface_hub._snapshot_download",
    "transformers",
    "sentence_transformers",
    "langchain",
    "langchain_community",
    "langchain_huggingface",
    "faiss",
    "httpx",
)


def setup_logging(
    name: str = "scinanoai",
    level: str | int | None = None,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configure and return a project logger.

    Idempotent: calling it twice with the same name returns the same instance
    without duplicating handlers. Routes a curated list of third-party
    loggers (HF Hub, transformers, sentence-transformers, langchain, faiss)
    through the same handler so model downloads and FAISS operations show up
    in container logs instead of vanishing into silence.
    """
    resolved_level = level or os.getenv("SCINANO_LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = resolved_level.upper()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(resolved_level)
        logger.addHandler(stream_handler)
        logger.propagate = False

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

    # Hoist third-party startup chatter into our handler.
    for lib_name in _NOISY_LIBRARIES:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(resolved_level)
        if not any(isinstance(h, logging.StreamHandler) for h in lib_logger.handlers):
            lib_logger.addHandler(stream_handler)
        lib_logger.propagate = False

    # Some libraries respect their own verbosity API on top of stdlib logging.
    try:
        import huggingface_hub.utils.logging as hf_logging

        hf_logging.set_verbosity_info()
    except Exception:  # noqa: BLE001 — optional, best effort
        pass
    try:
        import transformers

        transformers.logging.set_verbosity_info()
        transformers.logging.enable_progress_bar()
    except Exception:  # noqa: BLE001
        pass

    return logger
