"""Validation helpers for reconstructed test outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_saved_confusion_matrix(path: str | Path) -> np.ndarray:
    frame = pd.read_csv(path, index_col=0)
    matrix = frame.to_numpy(dtype=np.int64)
    if matrix.shape != (10, 10):
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        matrix = numeric.iloc[:, -10:].to_numpy(dtype=np.int64)
    if matrix.shape != (10, 10):
        raise ValueError(f"Expected a 10x10 confusion matrix in {path}; found {matrix.shape}.")
    return matrix
