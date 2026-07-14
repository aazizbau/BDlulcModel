"""Constants used by every spatial block bootstrap stage."""

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
