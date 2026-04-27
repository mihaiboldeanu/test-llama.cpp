"""Logging setup for rama."""

import logging


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup and return the main logger."""
    logger = logging.getLogger("rama")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"),
        )
        logger.addHandler(handler)

    return logger
