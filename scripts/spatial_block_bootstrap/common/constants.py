"""Constants used by every spatial block bootstrap stage.

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

    import scripts.spatial_block_bootstrap.common.constants
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = Path("outputs/spatial_block_bootstrap")
DEFAULT_BEST_RUNS_CSV = Path(
    "outputs/master_training_with_outputs/best_runs_by_group.csv"
)
DEFAULT_BOOTSTRAP_CONFIG = Path(
    "scripts/spatial_block_bootstrap/config/bootstrap_config.yaml"
)

CLASS_IDS = list(range(1, 11))
CLASS_NAMES = {
    1: "Urban / Institutional Built-up",
    2: "Rural Settlement (Homestead Vegetation)",
    3: "Transport & Coastal Embankments",
    4: "Cropland (All Crop Intensities)",
    5: "Tree-based Agroforestry & Orchard",
    6: "Aquaculture & Inland Ponds",
    7: "Canals & Drainage Network",
    8: "Rivers & Estuarine Channels",
    9: "Mangrove Forest",
    10: "Bare / Exposed Coastal Land",
}

MODEL_FAMILY_ORDER = [
    "CNN1D",
    "FTTransformer",
    "LightGBM",
    "MLP",
    "ResMLP",
    "XGBoost",
]
MODEL_FAMILY_MAP = {
    "cnn1d": "CNN1D",
    "fttransformer": "FTTransformer",
    "lgbm": "LightGBM",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
    "resmlp": "ResMLP",
    "xgboost": "XGBoost",
}
FEATURE_SET_ORDER = ["AE64", "AE64_plus10indices"]
FEATURE_SET_MAP = {
    "ae64": "AE64",
    "ae64_plus10indices": "AE64_plus10indices",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def family_display(value: str) -> str:
    key = str(value).lower().strip().replace("-", "").replace("_", "")
    aliases = {
        "cnn1d": "CNN1D",
        "fttransformer": "FTTransformer",
        "lightgbm": "LightGBM",
        "lgbm": "LightGBM",
        "mlp": "MLP",
        "resmlp": "ResMLP",
        "xgboost": "XGBoost",
        "xgb": "XGBoost",
    }
    return aliases.get(key, str(value))


def feature_display(value: str) -> str:
    key = str(value).lower().strip()
    return FEATURE_SET_MAP.get(key, str(value))
