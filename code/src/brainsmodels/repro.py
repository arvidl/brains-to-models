"""Pin RNGs so notebooks and figure scripts match on rerun."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 7


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy, and PYTHONHASHSEED. Return the seed used.

    ``PYTHONHASHSEED`` only affects a process if it is set before the
    interpreter starts (the Makefile exports it). Setting it here still
    documents the intended value and covers child processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
