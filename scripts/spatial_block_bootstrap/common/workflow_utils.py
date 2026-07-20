"""Load aligned inputs shared by bootstrap analysis scripts.

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

    import scripts.spatial_block_bootstrap.common.workflow_utils
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_OUTPUT_ROOT, resolve_path
from .output_utils import read_table


def load_bootstrap_context(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, list[str], np.ndarray, dict]:
    root = resolve_path(output_root)
    validation = json.loads(
        (root / "metadata" / "validation_status.json").read_text(encoding="utf-8")
    )
    if not validation.get("passed"):
        raise RuntimeError("Spatial block validation has not passed.")
    selected = pd.read_csv(root / "metadata" / "selected_runs.csv")
    block_long = read_table(
        root
        / "block_confusion_matrices"
        / "all_selected_runs_block_confusion_long.parquet"
    )
    block_ids = pd.read_csv(root / "bootstrap_indices" / "test_block_ids.csv")[
        "block_id"
    ].tolist()
    indices = np.load(root / "bootstrap_indices" / "shared_test_block_bootstrap_indices.npy")
    settings = json.loads(
        (root / "metadata" / "bootstrap_settings.json").read_text(encoding="utf-8")
    )
    return root, selected, block_long, block_ids, indices, settings
