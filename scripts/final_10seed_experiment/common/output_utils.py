"""Output helpers for final repeated-seed experiment scripts.

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

    import scripts.final_10seed_experiment.common.output_utils
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def copy_config(config_path: Path, output_root: Path, family: str) -> None:
    target = output_root / family / "config" / "selected_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, target)

