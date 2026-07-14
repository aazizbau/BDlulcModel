"""Constants for the final repeated-seed best-configuration experiment."""

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

