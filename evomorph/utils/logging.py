"""Project logger: one consistent format, configurable level via EVOMORPH_LOG_LEVEL."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def get_logger(name: str = "evomorph") -> logging.Logger:
    """Return a logger under the ``evomorph`` namespace, configuring the root once."""
    global _configured
    if not _configured:
        level = os.environ.get("EVOMORPH_LOG_LEVEL", "INFO").upper()
        root = logging.getLogger("evomorph")
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
            root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _configured = True
    if not name.startswith("evomorph"):
        name = f"evomorph.{name}"
    return logging.getLogger(name)
