"""Random-number helpers for reproducible shared resampling."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("Random seed must be non-negative.")
    return np.random.default_rng(seed)
