"""
Structured logging for MiniMind.  Wraps Python's ``logging`` with Rich
colourised console output, optional file output, and process-rank awareness
for distributed training.

Usage::

    from trainer.logger import get_logger
    log = get_logger(__name__)
    log.info("Training started")
    log.warning("GPU memory low")
    log.error("Checkpoint save failed", exc_info=True)

Backward-compatible shim — the global ``Logger()`` function still works::

    from trainer.logger import Logger       # deprecated alias
    Logger("Same as log.info(...)")
"""

import logging
import os
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

_LOG_FORMAT_VERBOSE = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_default_level = os.environ.get("MINIMIND_LOG_LEVEL", "INFO").upper()
_file_path = os.environ.get("MINIMIND_LOG_FILE", "")

_initialised = False
_root_logger: Optional[logging.Logger] = None


def _init_logging() -> logging.Logger:
    """Lazy-init the MiniMind root logger (idempotent)."""
    global _initialised, _root_logger
    if _initialised and _root_logger is not None:
        return _root_logger

    _root_logger = logging.getLogger("minimind")
    _root_logger.setLevel(getattr(logging, _default_level, logging.INFO))
    _root_logger.propagate = False

    # ---- console handler (Rich or plain) ----
    if _HAS_RICH:
        console = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        console.setLevel(logging.DEBUG)
    else:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(_LOG_FORMAT_VERBOSE))
    _root_logger.addHandler(console)

    # ---- optional file handler ----
    if _file_path:
        os.makedirs(os.path.dirname(_file_path) or ".", exist_ok=True)
        fh = logging.FileHandler(_file_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT_VERBOSE, datefmt="%Y-%m-%d %H:%M:%S"))
        _root_logger.addHandler(fh)

    _initialised = True
    return _root_logger


def get_logger(name: str = "minimind") -> logging.Logger:
    """Return a child logger under the minimind hierarchy.

    ``name`` is typically ``__name__`` from the calling module.
    """
    _init_logging()
    return _root_logger.getChild(name.split(".", 1)[1] if "." in name else name)  # type: ignore[union-attr]


def set_level(level: str) -> None:
    """Set the log level for the root minimind logger at runtime."""
    _init_logging()
    _root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))  # type: ignore[union-attr]


# ==========================================================================
# Backward-compatible global Logger() shim
# ==========================================================================

_global_log = None  # lazy


def Logger(content: str) -> None:
    """Deprecated shim that delegates to ``log.info()``.

    Kept for backward compatibility — all existing training scripts calling
    ``Logger(f"...")`` continue to work unchanged.
    """
    global _global_log
    if _global_log is None:
        _global_log = get_logger("minimind")
    _global_log.info(str(content))
