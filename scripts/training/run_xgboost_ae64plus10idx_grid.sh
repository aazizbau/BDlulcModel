#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# XGBoost grid search runner for AE64 + 10 indices classification
# ============================================================
#
# Runs 10 experiments and writes:
#   - per-run logs:
#       logs/xgb_ae64plus10idx_grid/<run_name>.log
#   - master comparison CSV:
#       runs/xgboost_ae64plus10idx_master_runs.csv
#
# Usage:
#   chmod +x scripts/training/run_xgboost_ae64plus10idx_grid.sh
#   ./scripts/training/run_xgboost_ae64plus10idx_grid.sh
#
# Notes:
# - Assumes you run from repo root.
# - Assumes your venv is already activated.
# - Skips a run if summary.json already exists in that run dir.
# - Rebuilds the master CSV from the known run list each time.
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA="data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz"
TRAIN_SCRIPT="scripts/training/train_xgboost_ae64_plus10indices_from_npz.py"

LOG_DIR="logs/xgb_ae64plus10idx_grid"
MASTER_CSV="runs/xgboost_ae64plus10idx_master_runs.csv"

mkdir -p "$LOG_DIR"
mkdir -p runs

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: data file not found: $DATA"
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: training script not found: $TRAIN_SCRIPT"
  exit 1
fi

run_one() {
  local run_name="$1"
  shift

  local outdir="runs/${run_name}"
  local logfile="${LOG_DIR}/${run_name}.log"

  echo "============================================================"
  echo "RUN: $run_name"
  echo "OUT: $outdir"
  echo "LOG: $logfile"
  echo "============================================================"

  if [[ -f "${outdir}/summary.json" ]]; then
    echo "Skipping ${run_name} because summary.json already exists."
    return 0
  fi

  python "$TRAIN_SCRIPT" \
    --data "$DATA" \
    --outdir "$outdir" \
    "$@" 2>&1 | tee "$logfile"
}

# ------------------------------------------------------------
# 10 candidate runs
# ------------------------------------------------------------
# Baseline observation from v1:
# - best iteration around 2189, far below 5000
# - AE64 + 10 indices clearly improves over AE64-only baseline
# - validation performance is strong, but train-vs-val gap is still clear
#
# Search logic:
# - lower/higher learning-rate around 0.03
# - shallower/deeper trees around max_depth 10
# - stronger min_child_weight on several runs
# - modest row/column subsampling changes
# - mild gamma / L1 / L2 regularization on some runs
# - because best iteration moved later than AE64-only, include a few longer runs
# ------------------------------------------------------------

run_one "xgb_ae64plus10idx_lr0.03_ne5000_md10_mcw3_sub0.8_col0.8_es200_v1" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --max-depth 10 \
  --min-child-weight 3.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --gamma 0.0 \
  --reg-alpha 0.0 \
  --reg-lambda 1.0 \
  --max-bin 256 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.02_ne7000_md10_mcw3_sub0.8_col0.8_es300_v2" \
  --learning-rate 0.02 \
  --n-estimators 7000 \
  --max-depth 10 \
  --min-child-weight 3.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --gamma 0.0 \
  --reg-alpha 0.0 \
  --reg-lambda 1.0 \
  --max-bin 256 \
  --early-stopping-rounds 300 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.03_ne5000_md8_mcw5_sub0.8_col0.8_es200_v3" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --max-depth 8 \
  --min-child-weight 5.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --gamma 0.0 \
  --reg-alpha 0.0 \
  --reg-lambda 1.5 \
  --max-bin 256 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.03_ne5000_md8_mcw8_sub0.8_col0.8_g0.1_ra0.0_rl2.0_es200_v4" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --max-depth 8 \
  --min-child-weight 8.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --gamma 0.1 \
  --reg-alpha 0.0 \
  --reg-lambda 2.0 \
  --max-bin 256 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.03_ne5000_md9_mcw5_sub0.7_col0.8_g0.1_ra0.1_rl2.0_es200_v5" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --max-depth 9 \
  --min-child-weight 5.0 \
  --subsample 0.7 \
  --colsample-bytree 0.8 \
  --gamma 0.1 \
  --reg-alpha 0.1 \
  --reg-lambda 2.0 \
  --max-bin 256 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.03_ne5000_md10_mcw5_sub0.7_col0.7_g0.2_ra0.1_rl2.0_es200_v6" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --max-depth 10 \
  --min-child-weight 5.0 \
  --subsample 0.7 \
  --colsample-bytree 0.7 \
  --gamma 0.2 \
  --reg-alpha 0.1 \
  --reg-lambda 2.0 \
  --max-bin 256 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.05_ne3500_md8_mcw5_sub0.8_col0.8_es150_v7" \
  --learning-rate 0.05 \
  --n-estimators 3500 \
  --max-depth 8 \
  --min-child-weight 5.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --gamma 0.0 \
  --reg-alpha 0.0 \
  --reg-lambda 1.5 \
  --max-bin 256 \
  --early-stopping-rounds 150 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.02_ne8000_md9_mcw8_sub0.8_col0.8_g0.1_ra0.1_rl2.0_es300_v8" \
  --learning-rate 0.02 \
  --n-estimators 8000 \
  --max-depth 9 \
  --min-child-weight 8.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --gamma 0.1 \
  --reg-alpha 0.1 \
  --reg-lambda 2.0 \
  --max-bin 256 \
  --early-stopping-rounds 300 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.03_ne5000_md12_mcw8_sub0.8_col0.8_g0.2_ra0.1_rl2.0_es200_v9" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --max-depth 12 \
  --min-child-weight 8.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --gamma 0.2 \
  --reg-alpha 0.1 \
  --reg-lambda 2.0 \
  --max-bin 256 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

run_one "xgb_ae64plus10idx_lr0.03_ne5000_md8_mcw10_sub0.9_col0.7_g0.2_ra0.5_rl3.0_es200_v10" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --max-depth 8 \
  --min-child-weight 10.0 \
  --subsample 0.9 \
  --colsample-bytree 0.7 \
  --gamma 0.2 \
  --reg-alpha 0.5 \
  --reg-lambda 3.0 \
  --max-bin 256 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

# ------------------------------------------------------------
# Rebuild master CSV from summary.json files
# ------------------------------------------------------------
python - <<'PY'
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

runs = [
    "xgb_ae64plus10idx_lr0.03_ne5000_md10_mcw3_sub0.8_col0.8_es200_v1",
    "xgb_ae64plus10idx_lr0.02_ne7000_md10_mcw3_sub0.8_col0.8_es300_v2",
    "xgb_ae64plus10idx_lr0.03_ne5000_md8_mcw5_sub0.8_col0.8_es200_v3",
    "xgb_ae64plus10idx_lr0.03_ne5000_md8_mcw8_sub0.8_col0.8_g0.1_ra0.0_rl2.0_es200_v4",
    "xgb_ae64plus10idx_lr0.03_ne5000_md9_mcw5_sub0.7_col0.8_g0.1_ra0.1_rl2.0_es200_v5",
    "xgb_ae64plus10idx_lr0.03_ne5000_md10_mcw5_sub0.7_col0.7_g0.2_ra0.1_rl2.0_es200_v6",
    "xgb_ae64plus10idx_lr0.05_ne3500_md8_mcw5_sub0.8_col0.8_es150_v7",
    "xgb_ae64plus10idx_lr0.02_ne8000_md9_mcw8_sub0.8_col0.8_g0.1_ra0.1_rl2.0_es300_v8",
    "xgb_ae64plus10idx_lr0.03_ne5000_md12_mcw8_sub0.8_col0.8_g0.2_ra0.1_rl2.0_es200_v9",
    "xgb_ae64plus10idx_lr0.03_ne5000_md8_mcw10_sub0.9_col0.7_g0.2_ra0.5_rl3.0_es200_v10",
]

master_csv = Path("runs/xgboost_ae64plus10idx_master_runs.csv")
master_csv.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "run_dir",
    "timestamp_jst",
    "data_npz",
    "model",
    "seed",
    "input_dim",
    "num_classes",
    "learning_rate",
    "n_estimators_requested",
    "best_iteration",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "max_bin",
    "early_stopping_rounds",
    "eval_every",
    "n_jobs",
    "tree_method",
    "grow_policy",
    "train_samples",
    "val_samples",
    "test_samples",
    "train_loss",
    "train_acc",
    "train_macro_f1",
    "train_balanced_acc",
    "best_val_loss",
    "best_val_acc",
    "best_val_macro_f1",
    "best_val_balanced_acc",
    "test_evaluated",
    "test_loss",
    "test_acc",
    "test_macro_f1",
    "test_balanced_acc",
    "total_train_seconds",
    "notes",
]

rows = []

for run_name in runs:
    run_dir = Path("runs") / run_name
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue

    with summary_path.open() as f:
        s = json.load(f)

    rows.append({
        "run_dir": str(run_dir),
        "timestamp_jst": s.get("created_at_jst", ""),
        "data_npz": s.get("data", ""),
        "model": "XGBoost",
        "seed": s.get("seed", ""),
        "input_dim": s.get("input_dim", ""),
        "num_classes": s.get("num_classes", ""),
        "learning_rate": s.get("learning_rate", ""),
        "n_estimators_requested": s.get("n_estimators_requested", ""),
        "best_iteration": s.get("best_iteration", ""),
        "max_depth": s.get("max_depth", ""),
        "min_child_weight": s.get("min_child_weight", ""),
        "subsample": s.get("subsample", ""),
        "colsample_bytree": s.get("colsample_bytree", ""),
        "gamma": s.get("gamma", ""),
        "reg_alpha": s.get("reg_alpha", ""),
        "reg_lambda": s.get("reg_lambda", ""),
        "max_bin": s.get("max_bin", ""),
        "early_stopping_rounds": s.get("early_stopping_rounds", ""),
        "eval_every": s.get("eval_every", ""),
        "n_jobs": s.get("n_jobs", ""),
        "tree_method": s.get("tree_method", ""),
        "grow_policy": s.get("grow_policy", ""),
        "train_samples": s.get("train_samples", ""),
        "val_samples": s.get("val_samples", ""),
        "test_samples": s.get("test_samples", ""),
        "train_loss": s.get("train_loss", ""),
        "train_acc": s.get("train_acc", ""),
        "train_macro_f1": s.get("train_macro_f1", ""),
        "train_balanced_acc": s.get("train_balanced_acc", ""),
        "best_val_loss": s.get("best_val_loss", ""),
        "best_val_acc": s.get("best_val_acc", ""),
        "best_val_macro_f1": s.get("best_val_macro_f1", ""),
        "best_val_balanced_acc": s.get("best_val_balanced_acc", ""),
        "test_evaluated": s.get("test_evaluated", ""),
        "test_loss": s.get("test_loss", ""),
        "test_acc": s.get("test_acc", ""),
        "test_macro_f1": s.get("test_macro_f1", ""),
        "test_balanced_acc": s.get("test_balanced_acc", ""),
        "total_train_seconds": s.get("total_train_seconds", ""),
        "notes": "",
    })

def num_or_inf(x, negative=False):
    if x in ("", None):
        return float("-inf") if negative else float("inf")
    try:
        return float(x)
    except Exception:
        return float("-inf") if negative else float("inf")

rows.sort(
    key=lambda r: (
        -num_or_inf(r["best_val_macro_f1"], negative=True),
        -num_or_inf(r["test_macro_f1"], negative=True),
        num_or_inf(r["best_val_loss"], negative=False),
    )
)

with master_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

ts = datetime.now(JST).isoformat(timespec="seconds")
print(f"[{ts}] Wrote master CSV: {master_csv}")
print(f"[{ts}] Rows: {len(rows)}")
PY

echo
echo "Done."
echo "Master CSV: $MASTER_CSV"
