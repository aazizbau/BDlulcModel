"""Build and load block-specific confusion matrices.

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

    import scripts.spatial_block_bootstrap.common.block_confusion_utils
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CLASS_IDS


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    n_classes = len(CLASS_IDS)
    true_zero = np.asarray(y_true, dtype=np.int64) - 1
    pred_zero = np.asarray(y_pred, dtype=np.int64) - 1
    valid = (
        (true_zero >= 0)
        & (true_zero < n_classes)
        & (pred_zero >= 0)
        & (pred_zero < n_classes)
    )
    flat = true_zero[valid] * n_classes + pred_zero[valid]
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes
    )


def predictions_to_block_long(predictions: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = ["run_name", "model_family", "model", "feature_set"]
    rows: list[dict[str, object]] = []
    metadata = predictions.iloc[0]
    for current_block, group in predictions.groupby("block_id", sort=True):
        cm = confusion_matrix(group["true_class_id"], group["pred_class_id"])
        for true_index, true_class in enumerate(CLASS_IDS):
            for pred_index, pred_class in enumerate(CLASS_IDS):
                row = {column: metadata[column] for column in metadata_cols}
                row.update(
                    {
                        "block_id": current_block,
                        "true_class_id": true_class,
                        "pred_class_id": pred_class,
                        "count": int(cm[true_index, pred_index]),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def block_tensor(
    block_long: pd.DataFrame,
    run_name: str,
    ordered_block_ids: list[str],
) -> np.ndarray:
    run = block_long[block_long["run_name"] == run_name]
    lookup = {value: index for index, value in enumerate(ordered_block_ids)}
    tensor = np.zeros((len(ordered_block_ids), len(CLASS_IDS), len(CLASS_IDS)), dtype=np.int64)
    for row in run.itertuples(index=False):
        if row.block_id in lookup:
            tensor[
                lookup[row.block_id],
                int(row.true_class_id) - 1,
                int(row.pred_class_id) - 1,
            ] = int(row.count)
    return tensor
