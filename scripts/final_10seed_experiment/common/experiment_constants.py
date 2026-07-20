"""Constants for the final repeated-seed best-configuration experiment.

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

    import scripts.final_10seed_experiment.common.experiment_constants
"""

from __future__ import annotations

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
CLASS_IDS = list(range(1, 11))
EXPERIMENT_NAME = "final_10seed_best_config"
DEFAULT_OUTPUT_ROOT = "outputs/final_10seed_experiment"
DEFAULT_BEST_RUNS_CSV = "outputs/master_training_with_outputs/best_runs_by_group.csv"

MODEL_FAMILY_ORDER = [
    "cnn1d",
    "fttransformer",
    "lgbm",
    "mlp",
    "resmlp",
    "xgboost",
]

MODEL_FAMILY_DISPLAY = {
    "cnn1d": "CNN1D",
    "fttransformer": "FTTransformer",
    "lgbm": "LightGBM",
    "mlp": "MLP",
    "resmlp": "ResMLP",
    "xgboost": "XGBoost",
}

TRAINING_SCRIPT_STEMS = {
    "cnn1d": "train_cnn1d",
    "fttransformer": "train_fttransformer",
    "lgbm": "train_lgbm",
    "mlp": "train_mlp",
    "resmlp": "train_resmlp",
    "xgboost": "train_xgboost",
}


def feature_set_suffix(feature_set: str) -> str:
    normalized = str(feature_set).lower().strip()
    if normalized == "ae64_plus10indices":
        return "ae64_plus10indices"
    if normalized == "ae64":
        return "ae64"
    raise ValueError(f"Unsupported feature set: {feature_set}")


def feature_set_for_run_name(feature_set: str) -> str:
    normalized = str(feature_set).lower().strip()
    if normalized == "ae64_plus10indices":
        return "AE64plus10indices"
    if normalized == "ae64":
        return "AE64"
    return str(feature_set).replace("_", "")

