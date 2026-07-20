"""Single authoritative implementation of confusion-matrix metrics.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Estimate confidence intervals by resampling the original spatial test blocks.

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
Regenerate block IDs, predictions, and selected-run metadata from the new AOI spatial split before resampling; never reuse this project's block inventory.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Import this helper from its parent workflow or an interactive check::

    import scripts.spatial_block_bootstrap.common.metric_utils
"""

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
