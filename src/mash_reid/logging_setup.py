"""Central logging setup: write run logs to a file for later analysis.

Call ``setup_logging()`` once at program start (CLI / GUI). It configures the
``mash_reid`` logger tree to write everything to a timestamped, rotating file
under ``logs/`` (UTF-8, so Thai paths/messages are preserved) and to echo a
chosen level to the console. Library modules just do
``log = logging.getLogger(__name__)`` and log normally.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_DIR = "logs"
_LOGGER_ROOT = "mash_reid"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor so callers don't import ``logging`` directly."""
    return logging.getLogger(name)


def setup_logging(log_dir: str = DEFAULT_LOG_DIR, console_level: int = logging.INFO,
                  console: bool = True) -> str:
    """Configure file + console logging and return the log file path.

    The file handler captures DEBUG and above (full detail for analysis); the
    console shows ``console_level`` and above. Safe to call more than once — it
    resets the ``mash_reid`` handlers each time.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"mash_reid_{ts}.log")

    logger = logging.getLogger(_LOGGER_ROOT)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("=== Mash-Object-AI run started, logging to %s ===", log_path)
    return log_path
