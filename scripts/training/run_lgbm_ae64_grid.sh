#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LightGBM grid search runner for AE64 classification
# ============================================================
#
# Runs 10 experiments and writes:
#   - per-run logs:
#       logs/lgbm_ae64_grid/<run_name>.log
#   - master comparison CSV:
#       runs/lgbm_ae64_master_runs.csv
#
# Usage:
#   chmod +x scripts/training/run_lgbm_ae64_grid.sh
#   ./scripts/training/run_lgbm_ae64_grid.sh
#
# Notes:
# - Assumes you run from repo root.
# - Assumes your venv is already activated.
# - Skips a run if summary.json already exists in that run dir.
# - Rebuilds the master CSV from the known run list each time.
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA="data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz"
TRAIN_SCRIPT="scripts/training/train_lgbm_ae64_from_npz.py"

LOG_DIR="logs/lgbm_ae64_grid"
MASTER_CSV="runs/lgbm_ae64_master_runs.csv"

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
# - best iteration around 745, far below 5000
# - validation performance is decent but there is a clear
#   train-vs-val gap, so some runs should reduce overfitting
#
# Search logic:
# - smaller/larger num_leaves around 127
# - stronger/weaker min_child_samples
# - learning-rate around 0.03
# - modest row/column subsampling changes
# - mild L1/L2 regularization on some runs
# ------------------------------------------------------------

run_one "lgbm_ae64_lr0.03_ne5000_nl127_mcs50_sub0.8_col0.8_es200_v1" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --num-leaves 127 \
  --min-child-samples 50 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.03_ne5000_nl63_mcs50_sub0.8_col0.8_es200_v2" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --num-leaves 63 \
  --min-child-samples 50 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.03_ne5000_nl255_mcs50_sub0.8_col0.8_es200_v3" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --num-leaves 255 \
  --min-child-samples 50 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.03_ne5000_nl127_mcs100_sub0.8_col0.8_es200_v4" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --num-leaves 127 \
  --min-child-samples 100 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.03_ne5000_nl127_mcs200_sub0.8_col0.8_es200_v5" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --num-leaves 127 \
  --min-child-samples 200 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.02_ne7000_nl127_mcs50_sub0.8_col0.8_es250_v6" \
  --learning-rate 0.02 \
  --n-estimators 7000 \
  --num-leaves 127 \
  --min-child-samples 50 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 250 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.05_ne4000_nl127_mcs50_sub0.8_col0.8_es150_v7" \
  --learning-rate 0.05 \
  --n-estimators 4000 \
  --num-leaves 127 \
  --min-child-samples 50 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 150 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.03_ne5000_nl127_mcs50_sub0.7_col0.7_es200_v8" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --num-leaves 127 \
  --min-child-samples 50 \
  --subsample 0.7 \
  --colsample-bytree 0.7 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.03_ne5000_nl127_mcs100_sub0.8_col0.8_ra0.1_rl0.5_es200_v9" \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --num-leaves 127 \
  --min-child-samples 100 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.1 \
  --reg-lambda 0.5 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

run_one "lgbm_ae64_lr0.02_ne7000_nl63_mcs100_sub0.7_col0.7_ra0.5_rl1.0_es250_v10" \
  --learning-rate 0.02 \
  --n-estimators 7000 \
  --num-leaves 63 \
  --min-child-samples 100 \
  --subsample 0.7 \
  --colsample-bytree 0.7 \
  --reg-alpha 0.5 \
  --reg-lambda 1.0 \
  --early-stopping-rounds 250 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
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
    "lgbm_ae64_lr0.03_ne5000_nl127_mcs50_sub0.8_col0.8_es200_v1",
    "lgbm_ae64_lr0.03_ne5000_nl63_mcs50_sub0.8_col0.8_es200_v2",
    "lgbm_ae64_lr0.03_ne5000_nl255_mcs50_sub0.8_col0.8_es200_v3",
    "lgbm_ae64_lr0.03_ne5000_nl127_mcs100_sub0.8_col0.8_es200_v4",
    "lgbm_ae64_lr0.03_ne5000_nl127_mcs200_sub0.8_col0.8_es200_v5",
    "lgbm_ae64_lr0.02_ne7000_nl127_mcs50_sub0.8_col0.8_es250_v6",
    "lgbm_ae64_lr0.05_ne4000_nl127_mcs50_sub0.8_col0.8_es150_v7",
    "lgbm_ae64_lr0.03_ne5000_nl127_mcs50_sub0.7_col0.7_es200_v8",
    "lgbm_ae64_lr0.03_ne5000_nl127_mcs100_sub0.8_col0.8_ra0.1_rl0.5_es200_v9",
    "lgbm_ae64_lr0.02_ne7000_nl63_mcs100_sub0.7_col0.7_ra0.5_rl1.0_es250_v10",
]

master_csv = Path("runs/lgbm_ae64_master_runs.csv")
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
        "model": "LightGBM",
        "seed": s.get("seed", ""),
        "input_dim": s.get("input_dim", ""),
        "num_classes": s.get("num_classes", ""),
        "learning_rate": s.get("learning_rate", ""),
        "n_estimators_requested": s.get("n_estimators_requested", ""),
        "best_iteration": s.get("best_iteration", ""),
        "num_leaves": s.get("num_leaves", ""),
        "max_depth": s.get("max_depth", ""),
        "min_child_samples": s.get("min_child_samples", ""),
        "subsample": s.get("subsample", ""),
        "subsample_freq": s.get("subsample_freq", ""),
        "colsample_bytree": s.get("colsample_bytree", ""),
        "min_split_gain": s.get("min_split_gain", ""),
        "reg_alpha": s.get("reg_alpha", ""),
        "reg_lambda": s.get("reg_lambda", ""),
        "max_bin": s.get("max_bin", ""),
        "early_stopping_rounds": s.get("early_stopping_rounds", ""),
        "eval_every": s.get("eval_every", ""),
        "n_jobs": s.get("n_jobs", ""),
        "force_col_wise": s.get("force_col_wise", ""),
        "force_row_wise": s.get("force_row_wise", ""),
        "deterministic": s.get("deterministic", ""),
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
