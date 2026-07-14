"""Metric utilities for final repeated-seed experiment outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .experiment_constants import CLASS_IDS


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def read_confusion_matrix(path: Path) -> np.ndarray:
    """Read an existing confusion-matrix CSV written by the training scripts."""
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > len(CLASS_IDS):
        numeric = numeric.iloc[:, -len(CLASS_IDS):]
    matrix = numeric.to_numpy(dtype=int)
    if matrix.shape != (len(CLASS_IDS), len(CLASS_IDS)):
        raise ValueError(f"Expected 10x10 confusion matrix in {path}, found {matrix.shape}")
    return matrix


def metrics_from_cm(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm)
    actual_total = cm.sum(axis=1)
    predicted_total = cm.sum(axis=0)
    total = cm.sum()

    producer = safe_divide(tp, actual_total)
    user = safe_divide(tp, predicted_total)
    f1 = safe_divide(2 * producer * user, producer + user)

    return {
        "overall_accuracy": float(safe_divide(tp.sum(), total).item()),
        "macro_producer_accuracy": float(np.mean(producer)),
        "macro_user_accuracy": float(np.mean(user)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(safe_divide(np.sum(f1 * actual_total), np.sum(actual_total)).item()),
        "total_support": int(total),
        "correct_count": int(tp.sum()),
    }


def confusion_matrix_to_long(cm: np.ndarray, metadata: dict[str, object]) -> pd.DataFrame:
    records = []
    for true_index, true_class in enumerate(CLASS_IDS):
        for pred_index, pred_class in enumerate(CLASS_IDS):
            records.append(
                {
                    **metadata,
                    "split": "test",
                    "true_class_id": true_class,
                    "pred_class_id": pred_class,
                    "count": int(cm[true_index, pred_index]),
                }
            )
    return pd.DataFrame(records)

