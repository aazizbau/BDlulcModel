#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Master runner for classification training experiments and output aggregation
# =============================================================================
#
# EXECUTION ORDER
# -----------------------------------------------------------------------------
# STAGE 1
#   Run all available classification grid shell scripts in sequence:
#     - MLP
#     - ResMLP
#     - 1D-CNN
#     - FT-Transformer
#     - LightGBM
#     - XGBoost
#   and across:
#     - AE64
#     - AE64 + 10 spectral indices
#
# STAGE 2
#   Standardize important run-level artifact filenames inside each run
#   directory so downstream aggregation can rely on canonical names:
#     - summary.json
#     - history.csv
#     - val_predictions.csv
#     - test_predictions.csv
#     - confusion_matrix_val.csv
#     - confusion_matrix_test.csv
#     - per_class_metrics_val.csv
#     - per_class_metrics_test.csv
#     - feature_importance_gain.csv
#
# STAGE 3
#   Scan master CSVs, summaries, histories, predictions, confusion matrices,
#   per-class metrics, feature-importance files, and source NPZ datasets to
#   build classification-oriented CSV outputs for tables and figures.
#
# -----------------------------------------------------------------------------
# PURPOSE
# -----------------------------------------------------------------------------
# This script creates a single analysis output structure for the Bangladesh
# coastal LULC project across:
#   - AE64 and AE64 + 10 indices
#   - MLP, ResMLP, 1D-CNN, FT-Transformer, LightGBM, XGBoost
#
# The generated CSVs are intended to support thesis/journal tables and figures
# related to:
#   - benchmark performance comparison
#   - AE64 vs AE64 + 10 indices ablation
#   - per-class metrics
#   - confusion matrices
#   - training curves
#   - uncertainty and confidence analysis
#   - dataset split summaries and class distributions
#
# -----------------------------------------------------------------------------
# MAIN OUTPUTS
# -----------------------------------------------------------------------------
#   outputs/master_training_with_outputs/script_run_registry.csv
#   outputs/master_training_with_outputs/standardized_run_files.csv
#   outputs/master_training_with_outputs/master_csv_inventory.csv
#   outputs/master_training_with_outputs/all_master_runs_long.csv
#   outputs/master_training_with_outputs/best_runs_by_group.csv
#   outputs/master_training_with_outputs/best_runs_overall.csv
#   outputs/master_training_with_outputs/all_summary_flat.csv
#   outputs/master_training_with_outputs/all_history_rows.csv
#   outputs/master_training_with_outputs/all_predictions_index.csv
#   outputs/master_training_with_outputs/all_predictions_combined.csv
#   outputs/master_training_with_outputs/all_confusion_matrices_long.csv
#   outputs/master_training_with_outputs/all_per_class_metrics_long.csv
#   outputs/master_training_with_outputs/all_feature_importance_long.csv
#   outputs/master_training_with_outputs/all_uncertainty_summary.csv
#   outputs/master_training_with_outputs/figure_4_12_uncertainty_error_bins.csv
#   outputs/master_training_with_outputs/figure_4_12_reliability_bins.csv
#   outputs/master_training_with_outputs/dataset_inventory.csv
#   outputs/master_training_with_outputs/table_2_4_input_feature_sets.csv
#   outputs/master_training_with_outputs/table_2_4_feature_names_long.csv
#   outputs/master_training_with_outputs/table_3_1_split_sample_summary.csv
#   outputs/master_training_with_outputs/table_3_2_class_distribution_across_splits.csv
#   outputs/master_training_with_outputs/table_3_3_model_configurations_and_tuning_ranges.csv
#   outputs/master_training_with_outputs/table_4_1_overall_test_performance_all_models.csv
#   outputs/master_training_with_outputs/table_4_2_per_class_metrics_best_model.csv
#   outputs/master_training_with_outputs/table_4_3_ablation_ae64_vs_ae64plus10idx.csv
#   outputs/master_training_with_outputs/table_A_1_full_hyperparameter_settings.csv
#   outputs/master_training_with_outputs/table_A_2_full_run_by_run_results.csv
#   outputs/master_training_with_outputs/figure_4_2_training_curves_long.csv
#   outputs/master_training_with_outputs/figure_4_3_model_comparison.csv
#   outputs/master_training_with_outputs/figure_4_4_ablation_source.csv
#   outputs/master_training_with_outputs/figure_4_5_best_model_confusion_matrix_long.csv
#   outputs/master_training_with_outputs/figure_4_6_best_model_per_class_metrics.csv
#   outputs/master_training_with_outputs/figure_A_7_tree_feature_importance_long.csv
#   outputs/master_training_with_outputs/artifact_inventory.csv
#
# -----------------------------------------------------------------------------
# NOTES
# -----------------------------------------------------------------------------
# - Runner scripts are expected under scripts/training/
# - The script continues to the next runner even if one fails
# - The script exits non-zero at the end if one or more runner scripts failed
# - Best-run selection prefers validation Macro F1 first, then test Macro F1,
#   then validation loss
# - Standardization copies matching files into canonical names when needed; it
#   does not delete original artifacts
# - This script only aggregates artifacts that currently exist in the repo; it
#   does not fabricate unavailable mapping/change-analysis outputs for 2017/2024
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   chmod +x shell_scripts/run_master_training_with_outputs.sh
#   ./shell_scripts/run_master_training_with_outputs.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUTROOT="outputs/master_training_with_outputs"
LOGDIR="${OUTROOT}/logs"
mkdir -p "$OUTROOT" "$LOGDIR"

MASTER_LOG="${LOGDIR}/master_training_with_outputs.log"
SCRIPT_REGISTRY_CSV="${OUTROOT}/script_run_registry.csv"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$MASTER_LOG"
}

SCRIPTS=(
  "scripts/training/run_mlp_ae64_grid.sh"
  "scripts/training/run_mlp_ae64plus10idx_grid.sh"
  "scripts/training/run_resmlp_ae64_grid.sh"
  "scripts/training/run_resmlp_ae64plus10idx_grid.sh"
  "scripts/training/run_cnn1d_ae64_grid.sh"
  "scripts/training/run_cnn1d_ae64plus10idx_grid.sh"
  "scripts/training/run_fttransformer_ae64_grid.sh"
  "scripts/training/run_fttransformer_ae64plus10idx_grid.sh"
  "scripts/training/run_lgbm_ae64_grid.sh"
  "scripts/training/run_lgbm_ae64plus10idx_grid.sh"
  "scripts/training/run_xgboost_ae64_grid.sh"
  "scripts/training/run_xgboost_ae64plus10idx_grid.sh"
)

echo "script_path,script_name,start_time,end_time,duration_seconds,status,log_file" > "$SCRIPT_REGISTRY_CSV"

log "Master classification training run started."
log "Project root: $PROJECT_ROOT"
log "Output root:  $OUTROOT"
log "Stage 1: run all training grid shell scripts first."
log "Stage 2: standardize run artifact filenames inside runs/."
log "Stage 3: aggregate classification artifacts for tables and figures."

FAILED_COUNT=0

# -----------------------------------------------------------------------------
# STAGE 1: RUN ALL TRAINING GRID SHELL SCRIPTS FIRST
# -----------------------------------------------------------------------------
for script_path in "${SCRIPTS[@]}"; do
  script_name="$(basename "$script_path")"
  script_stem="${script_name%.sh}"
  script_log="${LOGDIR}/${script_stem}.log"

  if [[ ! -f "$script_path" ]]; then
    log "ERROR: Missing script: $script_path"
    echo "\"$script_path\",\"$script_name\",\"\",\"\",0,\"missing\",\"$script_log\"" >> "$SCRIPT_REGISTRY_CSV"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if [[ ! -x "$script_path" ]]; then
    chmod +x "$script_path"
  fi

  start_iso="$(date --iso-8601=seconds)"
  start_epoch="$(date +%s)"

  log "Running training grid script: $script_path"
  set +e
  bash "$script_path" 2>&1 | tee "$script_log"
  exit_code=${PIPESTATUS[0]}
  set -e

  end_iso="$(date --iso-8601=seconds)"
  end_epoch="$(date +%s)"
  duration_sec=$((end_epoch - start_epoch))

  if [[ $exit_code -eq 0 ]]; then
    status="success"
    log "Completed successfully: $script_path (duration=${duration_sec}s)"
  else
    status="failed"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    log "FAILED: $script_path (exit_code=${exit_code}, duration=${duration_sec}s)"
  fi

  echo "\"$script_path\",\"$script_name\",\"$start_iso\",\"$end_iso\",\"$duration_sec\",\"$status\",\"$script_log\"" >> "$SCRIPT_REGISTRY_CSV"
done

log "All requested training grid shell scripts have been attempted."
log "Starting Stage 2: artifact standardization."

# -----------------------------------------------------------------------------
# STAGE 2: STANDARDIZE RUN-LEVEL FILENAMES INSIDE runs/
# -----------------------------------------------------------------------------
python3 - <<'PY'
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(".").resolve()
RUNS_DIR = PROJECT_ROOT / "runs"
OUTROOT = PROJECT_ROOT / "outputs" / "master_training_with_outputs"
OUTROOT.mkdir(parents=True, exist_ok=True)

STANDARDIZED_RUN_FILES = OUTROOT / "standardized_run_files.csv"

CANONICAL_ARTIFACTS = {
    "history_csv": "history.csv",
    "val_predictions_csv": "val_predictions.csv",
    "test_predictions_csv": "test_predictions.csv",
    "confusion_matrix_val_csv": "confusion_matrix_val.csv",
    "confusion_matrix_test_csv": "confusion_matrix_test.csv",
    "per_class_metrics_val_csv": "per_class_metrics_val.csv",
    "per_class_metrics_test_csv": "per_class_metrics_test.csv",
    "feature_importance_gain_csv": "feature_importance_gain.csv",
}


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def safe_load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def score_file(path: Path, canonical_name: str) -> int:
    name = path.name.lower()
    target = canonical_name.lower()
    score = 0
    if name == target:
        score += 1000
    if target.replace(".csv", "") in name:
        score += 500
    if "prediction" in target and "pred" in name:
        score += 200
    if name.endswith(".csv"):
        score += 100
    if name.endswith(".json") and target.endswith(".json"):
        score += 100
    return score


def choose_best(files: List[Path], canonical_name: str) -> Optional[Path]:
    if not files:
        return None
    ranked = sorted(files, key=lambda p: (score_file(p, canonical_name), p.name.lower()), reverse=True)
    best = ranked[0]
    return best if score_file(best, canonical_name) > 0 else None


def maybe_copy_to_canonical(run_dir: Path, src: Optional[Path], canonical_name: str) -> Tuple[str, str, str, str]:
    canonical_path = run_dir / canonical_name

    if src is None:
        return ("missing", "", rel(canonical_path), "no_candidate_found")

    if src.resolve() == canonical_path.resolve():
        return ("kept_existing", rel(src), rel(canonical_path), "already_canonical")

    try:
        shutil.copy2(src, canonical_path)
        return ("copied", rel(src), rel(canonical_path), "ok")
    except Exception as e:
        return ("copy_failed", rel(src), rel(canonical_path), str(e))


rows: List[Dict[str, str]] = []

if RUNS_DIR.exists():
    for run_dir in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        summary_path = run_dir / "summary.json"
        summary = safe_load_json(summary_path) if summary_path.exists() else None

        csv_files = sorted(run_dir.glob("*.csv"))
        json_files = sorted(run_dir.glob("*.json"))

        action, source_path, canonical_path, status = maybe_copy_to_canonical(
            run_dir,
            summary_path if summary_path.exists() else choose_best(json_files, "summary.json"),
            "summary.json",
        )
        rows.append(
            {
                "run_dir": rel(run_dir),
                "artifact_kind": "summary",
                "source_path": source_path,
                "canonical_path": canonical_path,
                "action": action,
                "status": status,
            }
        )

        for summary_key, canonical_name in CANONICAL_ARTIFACTS.items():
            src: Optional[Path] = None
            if isinstance(summary, dict):
                raw = summary.get(summary_key)
                if raw:
                    path = Path(str(raw))
                    if not path.is_absolute():
                        path = PROJECT_ROOT / path
                    if path.exists():
                        src = path

            if src is None:
                src = choose_best(csv_files, canonical_name)

            action, source_path, canonical_path, status = maybe_copy_to_canonical(run_dir, src, canonical_name)
            rows.append(
                {
                    "run_dir": rel(run_dir),
                    "artifact_kind": summary_key,
                    "source_path": source_path,
                    "canonical_path": canonical_path,
                    "action": action,
                    "status": status,
                }
            )

pd.DataFrame(rows).to_csv(STANDARDIZED_RUN_FILES, index=False)
print(f"Wrote: {STANDARDIZED_RUN_FILES}")
PY

log "Stage 2 finished."
log "Starting Stage 3: classification artifact aggregation."

# -----------------------------------------------------------------------------
# STAGE 3: AGGREGATE STANDARDIZED CLASSIFICATION OUTPUTS
# -----------------------------------------------------------------------------
python3 - <<'PY'
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()
RUNS_DIR = PROJECT_ROOT / "runs"
OUTROOT = PROJECT_ROOT / "outputs" / "master_training_with_outputs"
OUTROOT.mkdir(parents=True, exist_ok=True)

SCRIPT_REGISTRY_CSV = OUTROOT / "script_run_registry.csv"
STANDARDIZED_RUN_FILES = OUTROOT / "standardized_run_files.csv"
MASTER_CSV_INVENTORY = OUTROOT / "master_csv_inventory.csv"
ALL_MASTER_RUNS_LONG = OUTROOT / "all_master_runs_long.csv"
BEST_RUNS_BY_GROUP = OUTROOT / "best_runs_by_group.csv"
BEST_RUNS_OVERALL = OUTROOT / "best_runs_overall.csv"
ALL_SUMMARY_FLAT = OUTROOT / "all_summary_flat.csv"
ALL_HISTORY_ROWS = OUTROOT / "all_history_rows.csv"
ALL_PREDICTIONS_INDEX = OUTROOT / "all_predictions_index.csv"
ALL_PREDICTIONS_COMBINED = OUTROOT / "all_predictions_combined.csv"
ALL_CONFUSION_MATRICES_LONG = OUTROOT / "all_confusion_matrices_long.csv"
ALL_PER_CLASS_METRICS_LONG = OUTROOT / "all_per_class_metrics_long.csv"
ALL_FEATURE_IMPORTANCE_LONG = OUTROOT / "all_feature_importance_long.csv"
ALL_UNCERTAINTY_SUMMARY = OUTROOT / "all_uncertainty_summary.csv"
FIGURE_4_12_UNCERTAINTY_ERROR_BINS = OUTROOT / "figure_4_12_uncertainty_error_bins.csv"
FIGURE_4_12_RELIABILITY_BINS = OUTROOT / "figure_4_12_reliability_bins.csv"
DATASET_INVENTORY = OUTROOT / "dataset_inventory.csv"
TABLE_2_4_INPUT_FEATURE_SETS = OUTROOT / "table_2_4_input_feature_sets.csv"
TABLE_2_4_FEATURE_NAMES_LONG = OUTROOT / "table_2_4_feature_names_long.csv"
TABLE_3_1_SPLIT_SAMPLE_SUMMARY = OUTROOT / "table_3_1_split_sample_summary.csv"
TABLE_3_2_CLASS_DISTRIBUTION = OUTROOT / "table_3_2_class_distribution_across_splits.csv"
TABLE_3_3_MODEL_CONFIGS = OUTROOT / "table_3_3_model_configurations_and_tuning_ranges.csv"
TABLE_4_1_PERFORMANCE = OUTROOT / "table_4_1_overall_test_performance_all_models.csv"
TABLE_4_2_PER_CLASS_BEST = OUTROOT / "table_4_2_per_class_metrics_best_model.csv"
TABLE_4_3_ABLATION = OUTROOT / "table_4_3_ablation_ae64_vs_ae64plus10idx.csv"
TABLE_A_1_HPARAMS = OUTROOT / "table_A_1_full_hyperparameter_settings.csv"
TABLE_A_2_RUNS = OUTROOT / "table_A_2_full_run_by_run_results.csv"
FIGURE_4_2_TRAINING_CURVES = OUTROOT / "figure_4_2_training_curves_long.csv"
FIGURE_4_3_MODEL_COMPARISON = OUTROOT / "figure_4_3_model_comparison.csv"
FIGURE_4_4_ABLATION = OUTROOT / "figure_4_4_ablation_source.csv"
FIGURE_4_5_BEST_CONFUSION = OUTROOT / "figure_4_5_best_model_confusion_matrix_long.csv"
FIGURE_4_6_BEST_PER_CLASS = OUTROOT / "figure_4_6_best_model_per_class_metrics.csv"
FIGURE_A_7_TREE_IMPORTANCE = OUTROOT / "figure_A_7_tree_feature_importance_long.csv"
ARTIFACT_INVENTORY = OUTROOT / "artifact_inventory.csv"

MAX_COMBINED_PRED_ROWS = 1_000_000
NUM_CLASSES = 10

FAMILY_ORDER = ["mlp", "resmlp", "cnn1d", "fttransformer", "lgbm", "xgboost"]
DEEP_FAMILIES = {"mlp", "resmlp", "cnn1d", "fttransformer"}
TREE_FAMILIES = {"lgbm", "xgboost"}

HPARAM_COLUMNS = [
    "d_token",
    "n_blocks",
    "n_heads",
    "attention_dropout",
    "ff_dropout",
    "residual_dropout",
    "ff_multiplier",
    "grad_accum_steps",
    "effective_batch_size",
    "amp_enabled",
    "hidden_dims",
    "channels",
    "kernels",
    "head_dim",
    "dropout",
    "batch_size",
    "epochs_requested",
    "epochs_completed",
    "lr",
    "weight_decay",
    "label_smoothing",
    "scheduler",
    "scheduler_factor",
    "scheduler_patience",
    "learning_rate",
    "n_estimators_requested",
    "best_iteration",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "min_split_gain",
    "reg_alpha",
    "reg_lambda",
    "max_bin",
    "early_stopping_rounds",
    "eval_every",
    "n_jobs",
    "force_col_wise",
    "force_row_wise",
    "deterministic",
    "min_child_weight",
    "gamma",
    "tree_method",
    "grow_policy",
]


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_json(d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(d, list):
        items[parent_key] = json.dumps(d, ensure_ascii=False)
    else:
        items[parent_key] = d
    return items


def list_to_pipe(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


def parse_group_name(path: Path) -> str:
    return path.stem.replace("_master_runs", "")


def infer_feature_set(group_name: str) -> str:
    return "ae64_plus10indices" if "plus10idx" in group_name else "ae64"


def infer_model_family(group_name: str, model_type: str = "") -> str:
    if model_type:
        if model_type.startswith("fttransformer"):
            return "fttransformer"
        if model_type.startswith("resmlp"):
            return "resmlp"
        if model_type.startswith("mlp"):
            return "mlp"
        if model_type.startswith("cnn1d"):
            return "cnn1d"
        if model_type.startswith("lgbm"):
            return "lgbm"
        if model_type.startswith("xgboost"):
            return "xgboost"
    if group_name.startswith("fttransformer_") or group_name.startswith("ftt_"):
        return "fttransformer"
    if group_name.startswith("resmlp_"):
        return "resmlp"
    if group_name.startswith("mlp_"):
        return "mlp"
    if group_name.startswith("cnn1d_"):
        return "cnn1d"
    if group_name.startswith("lgbm_"):
        return "lgbm"
    if group_name.startswith("xgboost_") or group_name.startswith("xgb_"):
        return "xgboost"
    return ""


def num_or_inf(x: Any, negative: bool = False) -> float:
    if x in ("", None):
        return float("-inf") if negative else float("inf")
    try:
        return float(x)
    except Exception:
        return float("-inf") if negative else float("inf")


def pick_best_runs(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, gdf in df.groupby(group_cols, dropna=False):
        tmp = gdf.copy()
        tmp["best_val_macro_f1"] = pd.to_numeric(tmp["best_val_macro_f1"], errors="coerce")
        tmp["test_macro_f1"] = pd.to_numeric(tmp["test_macro_f1"], errors="coerce")
        tmp["best_val_loss"] = pd.to_numeric(tmp["best_val_loss"], errors="coerce")
        tmp = tmp.sort_values(
            by=["best_val_macro_f1", "test_macro_f1", "best_val_loss"],
            ascending=[False, False, True],
            na_position="last",
        )
        rows.append(tmp.head(1))
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def existing_paths_from_values(values: List[Any]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for value in values:
        if value in ("", None):
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            path = path.resolve()
        except Exception:
            pass
        if path.exists() and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def confidence_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("prob_class_")]


def compute_uncertainty_frames(df: pd.DataFrame, run_meta: Dict[str, Any]) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    prob_cols = confidence_cols(df)
    if not prob_cols or "y_true" not in df.columns or "y_pred" not in df.columns:
        return None, None, None

    tmp = df.copy()
    for c in ["y_true", "y_pred"] + prob_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.dropna(subset=["y_true", "y_pred"] + prob_cols)
    if tmp.empty:
        return None, None, None

    probs = tmp[prob_cols].to_numpy(dtype=float)
    top1 = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    top2 = sorted_probs[:, -2] if probs.shape[1] >= 2 else np.zeros_like(top1)
    margin = top1 - top2
    uncertainty = 1.0 - top1
    correct = (tmp["y_true"].to_numpy(dtype=int) == tmp["y_pred"].to_numpy(dtype=int)).astype(int)

    summary = dict(run_meta)
    summary.update(
        {
            "n_rows": int(len(tmp)),
            "accuracy": float(correct.mean()),
            "mean_confidence": float(top1.mean()),
            "mean_uncertainty": float(uncertainty.mean()),
            "mean_margin": float(margin.mean()),
            "error_rate": float(1.0 - correct.mean()),
        }
    )

    bin_edges = np.linspace(0.0, 1.0, 11)
    conf_bin = pd.cut(top1, bins=bin_edges, include_lowest=True, right=True)
    unc_bin = pd.cut(uncertainty, bins=bin_edges, include_lowest=True, right=True)

    reliability_df = (
        pd.DataFrame(
            {
                **run_meta,
                "confidence": top1,
                "correct": correct,
                "confidence_bin": conf_bin.astype(str),
            }
        )
        .groupby(["confidence_bin"], dropna=False)
        .agg(
            n_rows=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            empirical_accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    for k, v in run_meta.items():
        reliability_df[k] = v

    uncertainty_df = (
        pd.DataFrame(
            {
                **run_meta,
                "uncertainty": uncertainty,
                "correct": correct,
                "uncertainty_bin": unc_bin.astype(str),
            }
        )
        .groupby(["uncertainty_bin"], dropna=False)
        .agg(
            n_rows=("uncertainty", "size"),
            mean_uncertainty=("uncertainty", "mean"),
            error_rate=("correct", lambda x: float(1.0 - np.mean(x))),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    for k, v in run_meta.items():
        uncertainty_df[k] = v

    return reliability_df, uncertainty_df, summary


def parse_confusion_csv(path: Path, run_meta: Dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(path)
    true_col = df.columns[0]
    pred_cols = list(df.columns[1:])
    rows = []
    for _, row in df.iterrows():
        true_class = int(row[true_col])
        row_sum = float(row[pred_cols].sum())
        for pred_col in pred_cols:
            pred_class = int(pred_col)
            count = float(row[pred_col])
            rows.append(
                {
                    **run_meta,
                    "true_class_id": true_class,
                    "pred_class_id": pred_class,
                    "count": int(count),
                    "row_fraction": (count / row_sum) if row_sum > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def load_npz_inventory(data_path: Path, feature_set: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_rows: list[dict[str, Any]] = []
    feature_set_rows: list[dict[str, Any]] = []
    feature_name_rows: list[dict[str, Any]] = []
    class_distribution_rows: list[dict[str, Any]] = []

    try:
        with np.load(data_path, allow_pickle=True) as d:
            feature_names = [str(x) for x in d["feature_names"].tolist()]
            split_arrays = {
                "train": d["y_train"].astype(int),
                "val": d["y_val"].astype(int),
            }
            if "y_test" in d:
                split_arrays["test"] = d["y_test"].astype(int)
    except Exception:
        return inventory_rows, feature_set_rows, feature_name_rows, class_distribution_rows

    input_dim = len(feature_names)
    total_samples = int(sum(len(v) for v in split_arrays.values()))
    inventory_rows.append(
        {
            "data_npz": rel(data_path),
            "feature_set": feature_set,
            "input_dim": input_dim,
            "n_features": input_dim,
            "n_splits": len(split_arrays),
            "total_samples": total_samples,
            "has_test_split": "test" in split_arrays,
        }
    )

    ae64_names = feature_names[:64] if input_dim >= 64 else feature_names
    plus_names = feature_names[64:] if input_dim > 64 else []
    feature_set_rows.append(
        {
            "data_npz": rel(data_path),
            "feature_set": feature_set,
            "feature_group": "AE64_embeddings",
            "number_of_features": len(ae64_names),
            "feature_names": "|".join(ae64_names),
        }
    )
    if plus_names:
        feature_set_rows.append(
            {
                "data_npz": rel(data_path),
                "feature_set": feature_set,
                "feature_group": "spectral_indices",
                "number_of_features": len(plus_names),
                "feature_names": "|".join(plus_names),
            }
        )

    for idx, name in enumerate(feature_names, start=1):
        feature_name_rows.append(
            {
                "data_npz": rel(data_path),
                "feature_set": feature_set,
                "feature_index_1based": idx,
                "feature_name": name,
                "feature_group": "AE64_embeddings" if idx <= 64 else "spectral_indices",
            }
        )

    for split_name, y in split_arrays.items():
        split_total = int(len(y))
        for class_id in range(1, NUM_CLASSES + 1):
            count = int(np.sum(y == class_id))
            class_distribution_rows.append(
                {
                    "data_npz": rel(data_path),
                    "feature_set": feature_set,
                    "split": split_name,
                    "class_id": class_id,
                    "sample_count": count,
                    "percentage_within_split": (count / split_total) if split_total > 0 else 0.0,
                    "percentage_of_total_dataset": (count / total_samples) if total_samples > 0 else 0.0,
                }
            )

    return inventory_rows, feature_set_rows, feature_name_rows, class_distribution_rows


master_csvs = sorted(RUNS_DIR.glob("*master_runs.csv"))

artifact_rows: List[Dict[str, Any]] = []
for path in sorted(RUNS_DIR.glob("**/*")):
    if path.is_file():
        artifact_rows.append(
            {
                "artifact_type": path.suffix.lower().lstrip("."),
                "path": rel(path),
                "filename": path.name,
                "parent_dir": rel(path.parent),
                "size_bytes": path.stat().st_size,
            }
        )
pd.DataFrame(artifact_rows).to_csv(ARTIFACT_INVENTORY, index=False)

master_inventory_rows: List[Dict[str, Any]] = []
master_frames: List[pd.DataFrame] = []

for path in master_csvs:
    df = safe_read_csv(path)
    group_name = parse_group_name(path)
    if df is None:
        master_inventory_rows.append(
            {
                "master_csv": rel(path),
                "group_name": group_name,
                "n_rows": "",
                "n_cols": "",
                "status": "read_failed",
            }
        )
        continue

    df = df.copy()
    df["source_master_csv"] = rel(path)
    df["group_name"] = group_name
    if "run_dir" in df.columns:
        df["run_name"] = df["run_dir"].fillna("").astype(str).str.rstrip("/").str.split("/").str[-1]
    else:
        df["run_dir"] = ""
        df["run_name"] = ""
    if "model_type" not in df.columns:
        df["model_type"] = ""
    df["feature_set"] = df["group_name"].apply(infer_feature_set)
    df["model_family"] = [
        infer_model_family(g, mt) for g, mt in zip(df["group_name"], df["model_type"])
    ]
    if "summary_json" not in df.columns:
        df["summary_json"] = df["run_dir"].fillna("").astype(str) + "/summary.json"
    master_frames.append(df)
    master_inventory_rows.append(
        {
            "master_csv": rel(path),
            "group_name": group_name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "status": "ok",
        }
    )

pd.DataFrame(master_inventory_rows).to_csv(MASTER_CSV_INVENTORY, index=False)

all_master_df = pd.concat(master_frames, ignore_index=True, sort=False) if master_frames else pd.DataFrame()
if not all_master_df.empty:
    if "model" not in all_master_df.columns:
        all_master_df["model"] = ""
    all_master_df["family_rank"] = all_master_df["model_family"].apply(
        lambda x: FAMILY_ORDER.index(x) if x in FAMILY_ORDER else 999
    )
    all_master_df = all_master_df.sort_values(
        by=["feature_set", "family_rank", "group_name", "best_val_macro_f1"],
        ascending=[True, True, True, False],
        na_position="last",
    ).drop(columns=["family_rank"])
all_master_df.to_csv(ALL_MASTER_RUNS_LONG, index=False)

best_by_group_df = pick_best_runs(all_master_df, ["group_name"]) if not all_master_df.empty else pd.DataFrame()
best_by_group_df.to_csv(BEST_RUNS_BY_GROUP, index=False)

best_overall_df = pick_best_runs(all_master_df, ["feature_set"]) if not all_master_df.empty else pd.DataFrame()
if not best_overall_df.empty:
    best_overall_df = best_overall_df.sort_values(
        by=["best_val_macro_f1", "test_macro_f1", "best_val_loss"],
        ascending=[False, False, True],
        na_position="last",
    ).head(1)
best_overall_df.to_csv(BEST_RUNS_OVERALL, index=False)

summary_rows: List[Dict[str, Any]] = []
for summary_path in existing_paths_from_values(all_master_df["summary_json"].tolist() if not all_master_df.empty and "summary_json" in all_master_df.columns else []):
    data = safe_load_json(summary_path)
    if data is None:
        continue
    flat = flatten_json(data)
    flat["summary_json"] = rel(summary_path)
    flat["run_dir"] = rel(summary_path.parent)
    summary_rows.append(flat)
pd.DataFrame(summary_rows).to_csv(ALL_SUMMARY_FLAT, index=False)

history_frames: List[pd.DataFrame] = []
history_paths = existing_paths_from_values(all_master_df["history_csv"].tolist() if not all_master_df.empty and "history_csv" in all_master_df.columns else [])
for history_path in history_paths:
    df = safe_read_csv(history_path)
    if df is None:
        continue
    df = df.copy()
    df["history_csv"] = rel(history_path)
    df["run_dir"] = rel(history_path.parent)
    history_frames.append(df)
all_history_df = pd.concat(history_frames, ignore_index=True, sort=False) if history_frames else pd.DataFrame()
all_history_df.to_csv(ALL_HISTORY_ROWS, index=False)

prediction_frames: List[pd.DataFrame] = []
prediction_index_rows: List[Dict[str, Any]] = []
uncertainty_summary_rows: List[Dict[str, Any]] = []
uncertainty_bin_frames: List[pd.DataFrame] = []
reliability_bin_frames: List[pd.DataFrame] = []

prediction_paths = []
if not all_master_df.empty:
    for col, split in [("val_predictions_csv", "val"), ("test_predictions_csv", "test")]:
        if col in all_master_df.columns:
            for _, row in all_master_df.dropna(subset=[col]).iterrows():
                prediction_paths.append((Path(str(row[col])), row.to_dict(), split))

combined_rows = 0
for raw_path, row_meta, split in prediction_paths:
    path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
    if not path.exists():
        continue
    df = safe_read_csv(path)
    if df is None:
        continue

    run_meta = {
        "run_dir": row_meta.get("run_dir", ""),
        "run_name": row_meta.get("run_name", ""),
        "group_name": row_meta.get("group_name", ""),
        "model": row_meta.get("model", ""),
        "model_type": row_meta.get("model_type", ""),
        "model_family": row_meta.get("model_family", ""),
        "feature_set": row_meta.get("feature_set", ""),
        "split": split,
        "prediction_csv": rel(path),
    }

    prediction_index_rows.append(
        {
            **run_meta,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "has_probability_columns": len(confidence_cols(df)) > 0,
        }
    )

    reliability_df, uncertainty_df, summary = compute_uncertainty_frames(df, run_meta)
    if summary is not None:
        uncertainty_summary_rows.append(summary)
    if uncertainty_df is not None:
        uncertainty_bin_frames.append(uncertainty_df)
    if reliability_df is not None:
        reliability_bin_frames.append(reliability_df)

    if combined_rows < MAX_COMBINED_PRED_ROWS:
        tmp = df.copy()
        for k, v in run_meta.items():
            tmp[k] = v
        remaining = MAX_COMBINED_PRED_ROWS - combined_rows
        if len(tmp) > remaining:
            tmp = tmp.head(remaining)
        combined_rows += len(tmp)
        prediction_frames.append(tmp)

pd.DataFrame(prediction_index_rows).to_csv(ALL_PREDICTIONS_INDEX, index=False)
pd.concat(prediction_frames, ignore_index=True, sort=False).to_csv(ALL_PREDICTIONS_COMBINED, index=False) if prediction_frames else pd.DataFrame().to_csv(ALL_PREDICTIONS_COMBINED, index=False)
pd.DataFrame(uncertainty_summary_rows).to_csv(ALL_UNCERTAINTY_SUMMARY, index=False)
pd.concat(uncertainty_bin_frames, ignore_index=True, sort=False).to_csv(FIGURE_4_12_UNCERTAINTY_ERROR_BINS, index=False) if uncertainty_bin_frames else pd.DataFrame().to_csv(FIGURE_4_12_UNCERTAINTY_ERROR_BINS, index=False)
pd.concat(reliability_bin_frames, ignore_index=True, sort=False).to_csv(FIGURE_4_12_RELIABILITY_BINS, index=False) if reliability_bin_frames else pd.DataFrame().to_csv(FIGURE_4_12_RELIABILITY_BINS, index=False)

per_class_frames: List[pd.DataFrame] = []
confusion_frames: List[pd.DataFrame] = []
feature_importance_frames: List[pd.DataFrame] = []

if not all_master_df.empty:
    for _, row in all_master_df.iterrows():
        base_meta = {
            "run_dir": row.get("run_dir", ""),
            "run_name": row.get("run_name", ""),
            "group_name": row.get("group_name", ""),
            "model": row.get("model", ""),
            "model_type": row.get("model_type", ""),
            "model_family": row.get("model_family", ""),
            "feature_set": row.get("feature_set", ""),
        }

        for split, col in [("val", "per_class_metrics_val_csv"), ("test", "per_class_metrics_test_csv")]:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                path = Path(str(row[col]))
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                if path.exists():
                    df = safe_read_csv(path)
                    if df is not None:
                        df = df.copy()
                        for k, v in base_meta.items():
                            df[k] = v
                        df["split"] = split
                        df["per_class_metrics_csv"] = rel(path)
                        if "precision" in df.columns:
                            df["user_accuracy"] = df["precision"]
                        if "recall" in df.columns:
                            df["producer_accuracy"] = df["recall"]
                        per_class_frames.append(df)

        for split, col in [("val", "confusion_matrix_val_csv"), ("test", "confusion_matrix_test_csv")]:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                path = Path(str(row[col]))
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                if path.exists():
                    df = parse_confusion_csv(path, {**base_meta, "split": split, "confusion_matrix_csv": rel(path)})
                    confusion_frames.append(df)

        if "feature_importance_gain_csv" in row and pd.notna(row["feature_importance_gain_csv"]) and str(row["feature_importance_gain_csv"]).strip():
            path = Path(str(row["feature_importance_gain_csv"]))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                df = safe_read_csv(path)
                if df is not None:
                    df = df.copy()
                    for k, v in base_meta.items():
                        df[k] = v
                    df["feature_importance_gain_csv"] = rel(path)
                    feature_importance_frames.append(df)

all_per_class_df = pd.concat(per_class_frames, ignore_index=True, sort=False) if per_class_frames else pd.DataFrame()
all_per_class_df.to_csv(ALL_PER_CLASS_METRICS_LONG, index=False)

all_confusion_df = pd.concat(confusion_frames, ignore_index=True, sort=False) if confusion_frames else pd.DataFrame()
all_confusion_df.to_csv(ALL_CONFUSION_MATRICES_LONG, index=False)

all_feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True, sort=False) if feature_importance_frames else pd.DataFrame()
all_feature_importance_df.to_csv(ALL_FEATURE_IMPORTANCE_LONG, index=False)
all_feature_importance_df.to_csv(FIGURE_A_7_TREE_IMPORTANCE, index=False)

dataset_inventory_rows: List[Dict[str, Any]] = []
feature_set_rows: List[Dict[str, Any]] = []
feature_name_rows: List[Dict[str, Any]] = []
class_distribution_rows: List[Dict[str, Any]] = []
split_summary_rows: List[Dict[str, Any]] = []

seen_data_paths = set()
if not all_master_df.empty and "data_npz" in all_master_df.columns:
    for _, row in all_master_df[["data_npz", "feature_set"]].dropna().drop_duplicates().iterrows():
        raw = Path(str(row["data_npz"]))
        path = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
        if not path.exists():
            continue
        if path in seen_data_paths:
            continue
        seen_data_paths.add(path)

        inv_rows, feat_rows, feat_name_rows, dist_rows = load_npz_inventory(path, str(row["feature_set"]))
        dataset_inventory_rows.extend(inv_rows)
        feature_set_rows.extend(feat_rows)
        feature_name_rows.extend(feat_name_rows)
        class_distribution_rows.extend(dist_rows)

        try:
            with np.load(path, allow_pickle=True) as d:
                split_sizes = {
                    "train": int(len(d["y_train"])),
                    "val": int(len(d["y_val"])),
                }
                if "y_test" in d:
                    split_sizes["test"] = int(len(d["y_test"]))
        except Exception:
            split_sizes = {}

        total = int(sum(split_sizes.values()))
        for split_name, sample_count in split_sizes.items():
            split_summary_rows.append(
                {
                    "data_npz": rel(path),
                    "feature_set": str(row["feature_set"]),
                    "split": split_name,
                    "number_of_spatial_blocks": "",
                    "sample_count": sample_count,
                    "proportion_of_total_samples": (sample_count / total) if total > 0 else 0.0,
                }
            )

pd.DataFrame(dataset_inventory_rows).to_csv(DATASET_INVENTORY, index=False)
pd.DataFrame(feature_set_rows).to_csv(TABLE_2_4_INPUT_FEATURE_SETS, index=False)
pd.DataFrame(feature_name_rows).to_csv(TABLE_2_4_FEATURE_NAMES_LONG, index=False)
pd.DataFrame(split_summary_rows).to_csv(TABLE_3_1_SPLIT_SAMPLE_SUMMARY, index=False)
pd.DataFrame(class_distribution_rows).to_csv(TABLE_3_2_CLASS_DISTRIBUTION, index=False)

if not all_master_df.empty:
    config_rows = []
    for (model_family, feature_set), gdf in all_master_df.groupby(["model_family", "feature_set"], dropna=False):
        for col in HPARAM_COLUMNS:
            if col not in gdf.columns:
                continue
            values = sorted({str(v) for v in gdf[col].dropna().tolist() if str(v).strip() not in {"", "nan", "None"}})
            if not values:
                continue
            config_rows.append(
                {
                    "model_family": model_family,
                    "feature_set": feature_set,
                    "tuning_parameter": col,
                    "explored_values": "|".join(values),
                }
            )
    pd.DataFrame(config_rows).to_csv(TABLE_3_3_MODEL_CONFIGS, index=False)
else:
    pd.DataFrame().to_csv(TABLE_3_3_MODEL_CONFIGS, index=False)

if not best_by_group_df.empty:
    table_4_1 = best_by_group_df.copy()
    table_4_1["overall_accuracy"] = table_4_1.get("test_acc", "")
    table_4_1["weighted_f1"] = ""
    table_4_1["kappa"] = ""
    table_4_1["inference_time_seconds"] = ""
    keep_cols = [
        "run_dir",
        "run_name",
        "group_name",
        "model",
        "model_type",
        "model_family",
        "feature_set",
        "overall_accuracy",
        "best_val_macro_f1",
        "test_macro_f1",
        "weighted_f1",
        "kappa",
        "total_train_seconds",
        "inference_time_seconds",
        "best_epoch",
        "best_iteration",
        "best_val_loss",
        "test_loss",
    ]
    for col in keep_cols:
        if col not in table_4_1.columns:
            table_4_1[col] = ""
    table_4_1 = table_4_1[keep_cols].sort_values(["feature_set", "model_family"])
    table_4_1.to_csv(TABLE_4_1_PERFORMANCE, index=False)

    figure_4_3 = table_4_1.melt(
        id_vars=["run_dir", "run_name", "group_name", "model_family", "feature_set"],
        value_vars=["overall_accuracy", "best_val_macro_f1", "test_macro_f1"],
        var_name="metric_name",
        value_name="metric_value",
    )
    figure_4_3.to_csv(FIGURE_4_3_MODEL_COMPARISON, index=False)
else:
    pd.DataFrame().to_csv(TABLE_4_1_PERFORMANCE, index=False)
    pd.DataFrame().to_csv(FIGURE_4_3_MODEL_COMPARISON, index=False)

if not best_by_group_df.empty:
    ablation_rows = []
    best_by_family = pick_best_runs(best_by_group_df, ["model_family", "feature_set"])
    for model_family, fam_df in best_by_family.groupby("model_family", dropna=False):
        ae64 = fam_df[fam_df["feature_set"] == "ae64"].head(1)
        plus = fam_df[fam_df["feature_set"] == "ae64_plus10indices"].head(1)
        row = {"model_family": model_family}
        for prefix, src in [("ae64", ae64), ("ae64_plus10indices", plus)]:
            if src.empty:
                row[f"{prefix}_run_dir"] = ""
                row[f"{prefix}_overall_accuracy"] = ""
                row[f"{prefix}_test_macro_f1"] = ""
                row[f"{prefix}_best_val_macro_f1"] = ""
            else:
                s = src.iloc[0]
                row[f"{prefix}_run_dir"] = s.get("run_dir", "")
                row[f"{prefix}_overall_accuracy"] = s.get("test_acc", "")
                row[f"{prefix}_test_macro_f1"] = s.get("test_macro_f1", "")
                row[f"{prefix}_best_val_macro_f1"] = s.get("best_val_macro_f1", "")

        row["oa_improvement_plus10_minus_ae64"] = num_or_inf(row["ae64_plus10indices_overall_accuracy"], True) - num_or_inf(row["ae64_overall_accuracy"], True)
        row["macro_f1_improvement_plus10_minus_ae64"] = num_or_inf(row["ae64_plus10indices_test_macro_f1"], True) - num_or_inf(row["ae64_test_macro_f1"], True)
        ablation_rows.append(row)

    ablation_df = pd.DataFrame(ablation_rows).sort_values("model_family")
    ablation_df.to_csv(TABLE_4_3_ABLATION, index=False)
    ablation_df.melt(
        id_vars=["model_family"],
        value_vars=[
            "ae64_overall_accuracy",
            "ae64_plus10indices_overall_accuracy",
            "ae64_test_macro_f1",
            "ae64_plus10indices_test_macro_f1",
        ],
        var_name="metric_variant",
        value_name="metric_value",
    ).to_csv(FIGURE_4_4_ABLATION, index=False)
else:
    pd.DataFrame().to_csv(TABLE_4_3_ABLATION, index=False)
    pd.DataFrame().to_csv(FIGURE_4_4_ABLATION, index=False)

deep_best_df = pick_best_runs(best_by_group_df[best_by_group_df["model_family"].isin(DEEP_FAMILIES)], ["model_family", "feature_set"]) if not best_by_group_df.empty else pd.DataFrame()
if not deep_best_df.empty and not all_history_df.empty:
    selected = set(deep_best_df["run_dir"].tolist())
    fig_4_2 = all_history_df[all_history_df["run_dir"].isin(selected)].copy()
    fig_4_2.to_csv(FIGURE_4_2_TRAINING_CURVES, index=False)
else:
    pd.DataFrame().to_csv(FIGURE_4_2_TRAINING_CURVES, index=False)

best_model_row = best_overall_df.iloc[0].to_dict() if not best_overall_df.empty else None
if best_model_row is not None:
    best_run_dir = best_model_row.get("run_dir", "")
    best_per_class = all_per_class_df[
        (all_per_class_df.get("run_dir", pd.Series(dtype=str)) == best_run_dir)
        & (all_per_class_df.get("split", pd.Series(dtype=str)) == "test")
    ].copy() if not all_per_class_df.empty else pd.DataFrame()
    best_per_class.to_csv(TABLE_4_2_PER_CLASS_BEST, index=False)
    best_per_class.to_csv(FIGURE_4_6_BEST_PER_CLASS, index=False)

    best_confusion = all_confusion_df[
        (all_confusion_df.get("run_dir", pd.Series(dtype=str)) == best_run_dir)
        & (all_confusion_df.get("split", pd.Series(dtype=str)) == "test")
    ].copy() if not all_confusion_df.empty else pd.DataFrame()
    best_confusion.to_csv(FIGURE_4_5_BEST_CONFUSION, index=False)
else:
    pd.DataFrame().to_csv(TABLE_4_2_PER_CLASS_BEST, index=False)
    pd.DataFrame().to_csv(FIGURE_4_6_BEST_PER_CLASS, index=False)
    pd.DataFrame().to_csv(FIGURE_4_5_BEST_CONFUSION, index=False)

if not all_master_df.empty:
    table_a1_cols = [c for c in ["run_dir", "run_name", "group_name", "model", "model_type", "model_family", "feature_set"] + HPARAM_COLUMNS if c in all_master_df.columns]
    all_master_df[table_a1_cols].to_csv(TABLE_A_1_HPARAMS, index=False)
    all_master_df.to_csv(TABLE_A_2_RUNS, index=False)
else:
    pd.DataFrame().to_csv(TABLE_A_1_HPARAMS, index=False)
    pd.DataFrame().to_csv(TABLE_A_2_RUNS, index=False)

print(f"Wrote: {MASTER_CSV_INVENTORY}")
print(f"Wrote: {ALL_MASTER_RUNS_LONG}")
print(f"Wrote: {BEST_RUNS_BY_GROUP}")
print(f"Wrote: {BEST_RUNS_OVERALL}")
print(f"Wrote: {ALL_SUMMARY_FLAT}")
print(f"Wrote: {ALL_HISTORY_ROWS}")
print(f"Wrote: {ALL_PREDICTIONS_INDEX}")
print(f"Wrote: {ALL_PREDICTIONS_COMBINED}")
print(f"Wrote: {ALL_CONFUSION_MATRICES_LONG}")
print(f"Wrote: {ALL_PER_CLASS_METRICS_LONG}")
print(f"Wrote: {ALL_FEATURE_IMPORTANCE_LONG}")
print(f"Wrote: {ALL_UNCERTAINTY_SUMMARY}")
print(f"Wrote: {FIGURE_4_12_UNCERTAINTY_ERROR_BINS}")
print(f"Wrote: {FIGURE_4_12_RELIABILITY_BINS}")
print(f"Wrote: {DATASET_INVENTORY}")
print(f"Wrote: {TABLE_2_4_INPUT_FEATURE_SETS}")
print(f"Wrote: {TABLE_2_4_FEATURE_NAMES_LONG}")
print(f"Wrote: {TABLE_3_1_SPLIT_SAMPLE_SUMMARY}")
print(f"Wrote: {TABLE_3_2_CLASS_DISTRIBUTION}")
print(f"Wrote: {TABLE_3_3_MODEL_CONFIGS}")
print(f"Wrote: {TABLE_4_1_PERFORMANCE}")
print(f"Wrote: {TABLE_4_2_PER_CLASS_BEST}")
print(f"Wrote: {TABLE_4_3_ABLATION}")
print(f"Wrote: {TABLE_A_1_HPARAMS}")
print(f"Wrote: {TABLE_A_2_RUNS}")
print(f"Wrote: {FIGURE_4_2_TRAINING_CURVES}")
print(f"Wrote: {FIGURE_4_3_MODEL_COMPARISON}")
print(f"Wrote: {FIGURE_4_4_ABLATION}")
print(f"Wrote: {FIGURE_4_5_BEST_CONFUSION}")
print(f"Wrote: {FIGURE_4_6_BEST_PER_CLASS}")
print(f"Wrote: {FIGURE_A_7_TREE_IMPORTANCE}")
print(f"Wrote: {ARTIFACT_INVENTORY}")
PY

log "Classification artifact aggregation finished."
log "Important output CSVs:"
log "  - ${OUTROOT}/script_run_registry.csv"
log "  - ${OUTROOT}/standardized_run_files.csv"
log "  - ${OUTROOT}/master_csv_inventory.csv"
log "  - ${OUTROOT}/all_master_runs_long.csv"
log "  - ${OUTROOT}/best_runs_by_group.csv"
log "  - ${OUTROOT}/best_runs_overall.csv"
log "  - ${OUTROOT}/all_summary_flat.csv"
log "  - ${OUTROOT}/all_history_rows.csv"
log "  - ${OUTROOT}/all_predictions_index.csv"
log "  - ${OUTROOT}/all_predictions_combined.csv"
log "  - ${OUTROOT}/all_confusion_matrices_long.csv"
log "  - ${OUTROOT}/all_per_class_metrics_long.csv"
log "  - ${OUTROOT}/all_feature_importance_long.csv"
log "  - ${OUTROOT}/all_uncertainty_summary.csv"
log "  - ${OUTROOT}/figure_4_12_uncertainty_error_bins.csv"
log "  - ${OUTROOT}/figure_4_12_reliability_bins.csv"
log "  - ${OUTROOT}/table_2_4_input_feature_sets.csv"
log "  - ${OUTROOT}/table_3_1_split_sample_summary.csv"
log "  - ${OUTROOT}/table_3_2_class_distribution_across_splits.csv"
log "  - ${OUTROOT}/table_3_3_model_configurations_and_tuning_ranges.csv"
log "  - ${OUTROOT}/table_4_1_overall_test_performance_all_models.csv"
log "  - ${OUTROOT}/table_4_2_per_class_metrics_best_model.csv"
log "  - ${OUTROOT}/table_4_3_ablation_ae64_vs_ae64plus10idx.csv"
log "  - ${OUTROOT}/figure_4_2_training_curves_long.csv"
log "  - ${OUTROOT}/figure_4_3_model_comparison.csv"
log "  - ${OUTROOT}/figure_4_4_ablation_source.csv"
log "  - ${OUTROOT}/figure_4_5_best_model_confusion_matrix_long.csv"
log "  - ${OUTROOT}/figure_4_6_best_model_per_class_metrics.csv"
log "  - ${OUTROOT}/figure_A_7_tree_feature_importance_long.csv"
log "  - ${OUTROOT}/artifact_inventory.csv"

if [[ $FAILED_COUNT -gt 0 ]]; then
  log "Master training completed with ${FAILED_COUNT} failed script(s)."
  exit 1
fi

log "Master training completed successfully with all scripts passing."
