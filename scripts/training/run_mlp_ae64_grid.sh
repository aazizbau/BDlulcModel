#!/usr/bin/env bash
# ==============================================================================
# Reproduction and AOI adaptation
# ==============================================================================
# Purpose: Run mlp ae64 grid.
# Workflow role: Extract spatially split samples, train a classifier, or orchestrate hyperparameter experiments.
#
# Prerequisites:
#   1. Run from the repository root with the project environment activated.
#   2. Install requirements.txt and any system GDAL/Earth Engine dependencies.
#   3. Verify every input path and available disk/GPU resources before starting.
#
# AOI adaptation:
#   Replace NPZ/raster/vector inputs with samples extracted from the new AOI, preserve spatially disjoint splits, and review class IDs, feature order, block size, budgets, and random seeds.
#   Keep CRS, resolution, nodata, feature order, class IDs, and split metadata
#   consistent across all scripts invoked by this runner.
#
# Reproducible example:
#   bash scripts/training/run_mlp_ae64_grid.sh
#
# Outputs and logs are controlled by the variables below. Use a new output/log
# location for a new AOI, retain the run manifest, and inspect failures before
# resuming. Existing usage notes and worked commands below remain authoritative.
# ==============================================================================
set -euo pipefail

# ============================================================
# MLP grid search runner for AE64 classification
# ============================================================
#
# Runs 10 experiments and writes:
#   - per-run logs:
#       logs/mlp_ae64_grid/<run_name>.log
#   - master comparison CSV:
#       runs/mlp_ae64_master_runs.csv
#
# Usage:
#   chmod +x scripts/training/run_mlp_ae64_grid.sh
#   ./scripts/training/run_mlp_ae64_grid.sh
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
TRAIN_SCRIPT="scripts/training/train_mlp_ae64_from_npz.py"

LOG_DIR="logs/mlp_ae64_grid"
MASTER_CSV="runs/mlp_ae64_master_runs.csv"

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
# Baseline:
# v1 = your current strong baseline
#
# Search logic:
# - test slightly larger and slightly smaller capacity
# - test slightly higher/lower dropout
# - test slightly lower/higher LR around 1e-3
# - test batch 4096 vs 8192
# - test small label smoothing changes
# - keep scheduler enabled because it clearly helped
# ------------------------------------------------------------

run_one "mlp_ae64_h512-256_do02_lr1e3_bs4096_v1" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h512-256_do03_lr1e3_bs4096_v2" \
  --hidden 512 256 \
  --dropout 0.3 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h512-256_do015_lr1e3_bs4096_v3" \
  --hidden 512 256 \
  --dropout 0.15 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h768-384_do02_lr1e3_bs4096_v4" \
  --hidden 768 384 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h1024-512_do03_lr1e3_bs4096_v5" \
  --hidden 1024 512 \
  --dropout 0.3 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h512-256_do02_lr7e4_bs4096_v6" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 7e-4 \
  --weight-decay 1e-4 \
  --patience 18 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 6 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h512-256_do02_lr15e3_bs4096_v7" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1.5e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h512-256_do02_lr1e3_bs8192_v8" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 8192 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h512-256_do02_lr1e3_bs4096_wd5e5_ls003_v9" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 5e-5 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.03 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64_h768-256_do025_lr1e3_bs4096_wd2e4_ls008_v10" \
  --hidden 768 256 \
  --dropout 0.25 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 2e-4 \
  --patience 18 \
  --min-delta 1e-4 \
  --label-smoothing 0.08 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 6 \
  --eval-every 1 \
  --device cuda \
  --seed 42

# ------------------------------------------------------------
# Rebuild master CSV from summary.json files
# ------------------------------------------------------------
python scripts/training/build_master_csv.py \
  --family mlp \
  --master-csv "$MASTER_CSV" \
  --runs \
  "mlp_ae64_h512-256_do02_lr1e3_bs4096_v1" \
  "mlp_ae64_h512-256_do03_lr1e3_bs4096_v2" \
  "mlp_ae64_h512-256_do015_lr1e3_bs4096_v3" \
  "mlp_ae64_h768-384_do02_lr1e3_bs4096_v4" \
  "mlp_ae64_h1024-512_do03_lr1e3_bs4096_v5" \
  "mlp_ae64_h512-256_do02_lr7e4_bs4096_v6" \
  "mlp_ae64_h512-256_do02_lr15e3_bs4096_v7" \
  "mlp_ae64_h512-256_do02_lr1e3_bs8192_v8" \
  "mlp_ae64_h512-256_do02_lr1e3_bs4096_wd5e5_ls003_v9" \
  "mlp_ae64_h768-256_do025_lr1e3_bs4096_wd2e4_ls008_v10"

if false; then
python - <<'PY'
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

runs = [
    "mlp_ae64_h512-256_do02_lr1e3_bs4096_v1",
    "mlp_ae64_h512-256_do03_lr1e3_bs4096_v2",
    "mlp_ae64_h512-256_do015_lr1e3_bs4096_v3",
    "mlp_ae64_h768-384_do02_lr1e3_bs4096_v4",
    "mlp_ae64_h1024-512_do03_lr1e3_bs4096_v5",
    "mlp_ae64_h512-256_do02_lr7e4_bs4096_v6",
    "mlp_ae64_h512-256_do02_lr15e3_bs4096_v7",
    "mlp_ae64_h512-256_do02_lr1e3_bs8192_v8",
    "mlp_ae64_h512-256_do02_lr1e3_bs4096_wd5e5_ls003_v9",
    "mlp_ae64_h768-256_do025_lr1e3_bs4096_wd2e4_ls008_v10",
]

master_csv = Path("runs/mlp_ae64_master_runs.csv")
master_csv.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "run_dir",
    "timestamp_jst",
    "data_npz",
    "model",
    "seed",
    "input_dim",
    "num_classes",
    "hidden_dims",
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
    "train_samples",
    "val_samples",
    "test_samples",
    "best_epoch",
    "best_val_macro_f1",
    "best_val_loss",
    "best_val_acc",
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
        "model": "MLPClassifier",
        "seed": s.get("seed", ""),
        "input_dim": s.get("input_dim", ""),
        "num_classes": s.get("num_classes", ""),
        "hidden_dims": "-".join(str(x) for x in s.get("hidden_dims", [])),
        "dropout": s.get("dropout", ""),
        "batch_size": s.get("batch_size", ""),
        "epochs_requested": s.get("epochs_requested", ""),
        "epochs_completed": s.get("epochs_completed", ""),
        "lr": s.get("lr", ""),
        "weight_decay": s.get("weight_decay", ""),
        "label_smoothing": s.get("label_smoothing", ""),
        "scheduler": s.get("scheduler", ""),
        "scheduler_factor": s.get("scheduler_factor", ""),
        "scheduler_patience": s.get("scheduler_patience", ""),
        "train_samples": s.get("train_samples", ""),
        "val_samples": s.get("val_samples", ""),
        "test_samples": s.get("test_samples", ""),
        "best_epoch": s.get("best_epoch", ""),
        "best_val_macro_f1": s.get("best_val_macro_f1", ""),
        "best_val_loss": s.get("best_val_loss", ""),
        "best_val_acc": s.get("best_val_acc", ""),
        "best_val_balanced_acc": s.get("best_val_balanced_acc", ""),
        "test_evaluated": s.get("test_evaluated", ""),
        "test_loss": s.get("test_loss", ""),
        "test_acc": s.get("test_acc", ""),
        "test_macro_f1": s.get("test_macro_f1", ""),
        "test_balanced_acc": s.get("test_balanced_acc", ""),
        "total_train_seconds": s.get("total_train_seconds", ""),
        "notes": "",
    })

rows.sort(
    key=lambda r: (
        -(float(r["best_val_macro_f1"]) if r["best_val_macro_f1"] not in ("", None) else float("-inf")),
        -(float(r["test_macro_f1"]) if r["test_macro_f1"] not in ("", None) else float("-inf")),
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
fi

echo
echo "Done."
echo "Master CSV: $MASTER_CSV"
