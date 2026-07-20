"""Metric utilities for final repeated-seed experiment outputs.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Run or summarize repeated-seed experiments for empirical model-performance uncertainty.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is an imported helper module rather than a standalone command. Its public functions are exercised by the parent workflow scripts.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate selected configurations and seed outputs from the new AOI training data; do not mix summaries from different spatial splits.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Import this helper from its parent workflow or an interactive check::

    import scripts.final_10seed_experiment.common.metric_utils
"""

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

