"""Validation helpers for reconstructed test outputs.

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

    import scripts.spatial_block_bootstrap.common.validation_utils
"""

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
