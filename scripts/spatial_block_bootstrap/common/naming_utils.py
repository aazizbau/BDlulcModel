"""Stable file and identifier naming helpers.

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

    import scripts.spatial_block_bootstrap.common.naming_utils
"""

from __future__ import annotations

import re


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def block_id(upazila: str, block_row: int, block_col: int) -> str:
    return f"{upazila}:r{block_row:04d}_c{block_col:04d}"
