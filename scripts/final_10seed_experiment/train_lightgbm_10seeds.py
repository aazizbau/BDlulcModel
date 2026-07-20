#!/usr/bin/env python3
"""
Run the final 10-seed experiment for the selected best LightGBM configuration.

Complete Example Run
--------------------
python scripts/final_10seed_experiment/train_lightgbm_10seeds.py \
    --config scripts/final_10seed_experiment/configs/lgbm_best.yaml \
    --output-root outputs/final_10seed_experiment

Reproduction and AOI adaptation
-------------------------------
Workflow role: Run or summarize repeated-seed experiments for empirical model-performance uncertainty.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This script has no formal argparse interface; review the documented path constants near the top of the file before execution.
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
"""

from common.runner_utils import run_family


if __name__ == "__main__":
    run_family("lgbm")

