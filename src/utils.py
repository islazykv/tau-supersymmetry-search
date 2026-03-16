from __future__ import annotations

import logging
import warnings

log = logging.getLogger(__name__)


def suppress_warnings():
    """Suppress all Python runtime warnings."""
    warnings.filterwarnings("ignore")
    log.info("Unessential warnings suppressed.")
