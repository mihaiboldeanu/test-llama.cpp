"""Logging setup for rama."""

import logging
import os


def setup_logging(verbose: bool | None = None) -> logging.Logger:
    """Setup and return the main logger."""
    if verbose is None:
        verbose = os.environ.get("RAMA_VERBOSE", "").lower() in ("1", "true", "yes")
    logger = logging.getLogger("rama")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"),
        )
        logger.addHandler(handler)

    return logger
