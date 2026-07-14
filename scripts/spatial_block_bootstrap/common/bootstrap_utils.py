"""Paired spatial block bootstrap computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metric_utils import metrics_from_cm, scalar_metric_rows


def generate_indices(n_blocks: int, replicates: int, seed: int) -> np.ndarray:
    if n_blocks <= 0:
        raise ValueError("At least one test block is required.")
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be greater than zero.")
    rng = np.random.default_rng(seed)
    dtype = np.uint16 if n_blocks <= np.iinfo(np.uint16).max else np.uint32
    return rng.integers(0, n_blocks, size=(replicates, n_blocks), dtype=dtype)


def indices_to_weights(indices: np.ndarray, n_blocks: int) -> np.ndarray:
    weights = np.zeros((indices.shape[0], n_blocks), dtype=np.uint16)
    for replicate, sampled in enumerate(indices):
        weights[replicate] = np.bincount(sampled, minlength=n_blocks)
    return weights


def aggregate_bootstrap_cms(
    block_confusions: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> np.ndarray:
    if block_confusions.shape[0] != bootstrap_indices.shape[1]:
        raise ValueError("Block tensor and shared bootstrap indices are not aligned.")
    weights = indices_to_weights(bootstrap_indices, block_confusions.shape[0])
    flat = block_confusions.reshape(block_confusions.shape[0], -1)
    return (weights @ flat).reshape(-1, block_confusions.shape[1], block_confusions.shape[2])


def scalar_distribution(
    block_confusions: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for replicate, cm in enumerate(
        aggregate_bootstrap_cms(block_confusions, bootstrap_indices)
    ):
        for metric, value in scalar_metric_rows(metrics_from_cm(cm)).items():
            rows.append(
                {"replicate": replicate, "metric": metric, "value": value * 100.0}
            )
    return pd.DataFrame(rows)


def percentile_summary(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return {
            "bootstrap_mean": np.nan,
            "lower_95": np.nan,
            "upper_95": np.nan,
            "valid_replicates": 0,
            "invalid_replicates": int(values.size),
        }
    return {
        "bootstrap_mean": float(np.nanmean(values)),
        "lower_95": float(np.nanpercentile(values, lower)),
        "upper_95": float(np.nanpercentile(values, upper)),
        "valid_replicates": int(valid.sum()),
        "invalid_replicates": int((~valid).sum()),
    }
