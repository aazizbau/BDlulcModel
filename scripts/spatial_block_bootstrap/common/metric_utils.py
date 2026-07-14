"""Single authoritative implementation of confusion-matrix metrics."""

from __future__ import annotations

import numpy as np


def _divide_nan(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def metrics_from_cm(cm: np.ndarray) -> dict[str, object]:
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square; found {cm.shape}.")

    tp = np.diag(cm)
    actual = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    total = float(cm.sum())

    producer = _divide_nan(tp, actual)
    user = _divide_nan(tp, predicted)
    f1 = _divide_nan(2.0 * producer * user, producer + user)

    overall = float(tp.sum() / total) if total else np.nan
    weights = actual / total if total else np.full(actual.shape, np.nan)

    return {
        "overall_accuracy": overall,
        "producer_accuracy": producer,
        "user_accuracy": user,
        "f1_score": f1,
        "macro_producer_accuracy": float(np.nanmean(producer)),
        "macro_user_accuracy": float(np.nanmean(user)),
        "macro_f1": float(np.nanmean(f1)),
        "weighted_producer_accuracy": float(np.nansum(producer * weights)),
        "weighted_user_accuracy": float(np.nansum(user * weights)),
        "weighted_f1": float(np.nansum(f1 * weights)),
        "actual_total": actual,
        "predicted_total": predicted,
        "correct_by_class": tp,
        "total_support": int(total),
        "correct_count": int(tp.sum()),
    }


def scalar_metric_rows(metrics: dict[str, object]) -> dict[str, float]:
    return {
        "Overall Accuracy": float(metrics["overall_accuracy"]),
        "Macro F1-score": float(metrics["macro_f1"]),
        "Weighted F1-score": float(metrics["weighted_f1"]),
    }
