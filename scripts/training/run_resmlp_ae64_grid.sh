#!/usr/bin/env bash
# ==============================================================================
# Reproduction and AOI adaptation
# ==============================================================================
# Purpose: Run resmlp ae64 grid.
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
#   bash scripts/training/run_resmlp_ae64_grid.sh
#
# Outputs and logs are controlled by the variables below. Use a new output/log
# location for a new AOI, retain the run manifest, and inspect failures before
# resuming. Existing usage notes and worked commands below remain authoritative.
# ==============================================================================
set -euo pipefail

# ============================================================
# ResMLP grid search runner for AE64 classification
# ============================================================
#
# Runs 10 experiments and writes:
#   - per-run logs:
#       logs/resmlp_ae64_grid/<run_name>.log
#   - master comparison CSV:
#       runs/resmlp_ae64_master_runs.csv
#
# Usage:
#   chmod +x scripts/training/run_resmlp_ae64_grid.sh
#   ./scripts/training/run_resmlp_ae64_grid.sh
#
# Notes:
# - Assumes you run from repo root.
# - Assumes your venv is already activated.
# - Skips a run if summary.json already exists in that run dir.
# - Rebuilds the master CSV from the known run list each time.
# - Search is centered on the first run behavior:
#     * best validation performance arrived very early (epoch 5)
#     * training continued improving while validation degraded
#     * therefore several candidates increase regularization and/or
#       reduce learning rate instead of only increasing capacity
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA="data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz"
TRAIN_SCRIPT="scripts/training/train_resmlp_ae64_from_npz.py"
LOG_DIR="logs/resmlp_ae64_grid"
MASTER_CSV="runs/resmlp_ae64_master_runs.csv"

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
# - best val macro F1 = 0.7305 at epoch 5
# - scheduler reduced lr later, but validation did not recover
# - train metrics kept climbing while val metrics flattened/dropped
# - likely mild overfitting and learning-rate a little aggressive
#
# Search logic:
# - keep baseline as v1
# - try slightly lower lr around 1e-3
# - try stronger weight decay / dropout / label smoothing
# - try smaller and larger residual widths
# - try 3-block residual stacks for depth
# - vary batch size modestly
# ------------------------------------------------------------

run_one "resmlp_ae64_h512-256_do02_lr1e3_bs4096_v1" \
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

run_one "resmlp_ae64_h512-256_do03_lr7e4_bs4096_v2" \
  --hidden 512 256 \
  --dropout 0.3 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 7e-4 \
  --weight-decay 2e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.08 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 4 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "resmlp_ae64_h512-256_do025_lr5e4_bs4096_v3" \
  --hidden 512 256 \
  --dropout 0.25 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 5e-4 \
  --weight-decay 3e-4 \
  --patience 18 \
  --min-delta 1e-4 \
  --label-smoothing 0.08 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 4 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "resmlp_ae64_h256-256_do02_lr1e3_bs4096_v4" \
  --hidden 256 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 2e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "resmlp_ae64_h256-256-256_do025_lr7e4_bs4096_v5" \
  --hidden 256 256 256 \
  --dropout 0.25 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 7e-4 \
  --weight-decay 3e-4 \
  --patience 18 \
  --min-delta 1e-4 \
  --label-smoothing 0.08 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 4 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "resmlp_ae64_h512-512-256_do03_lr7e4_bs4096_v6" \
  --hidden 512 512 256 \
  --dropout 0.3 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 7e-4 \
  --weight-decay 3e-4 \
  --patience 18 \
  --min-delta 1e-4 \
  --label-smoothing 0.08 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 4 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "resmlp_ae64_h768-384_do03_lr7e4_bs4096_v7" \
  --hidden 768 384 \
  --dropout 0.3 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 7e-4 \
  --weight-decay 3e-4 \
  --patience 18 \
  --min-delta 1e-4 \
  --label-smoothing 0.10 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 4 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "resmlp_ae64_h512-256_do025_lr7e4_bs2048_v8" \
  --hidden 512 256 \
  --dropout 0.25 \
  --batch-size 2048 \
  --epochs 100 \
  --lr 7e-4 \
  --weight-decay 2e-4 \
  --patience 18 \
  --min-delta 1e-4 \
  --label-smoothing 0.08 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 4 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "resmlp_ae64_h512-256_do02_lr8e4_bs8192_v9" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 8192 \
  --epochs 100 \
  --lr 8e-4 \
  --weight-decay 2e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 4 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "resmlp_ae64_h384-384-192_do03_lr5e4_bs4096_v10" \
  --hidden 384 384 192 \
  --dropout 0.3 \
  --batch-size 4096 \
  --epochs 120 \
  --lr 5e-4 \
  --weight-decay 5e-4 \
  --patience 20 \
  --min-delta 1e-4 \
  --label-smoothing 0.10 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 4 \
  --eval-every 1 \
  --device cuda \
  --seed 42

# ------------------------------------------------------------
# Rebuild master CSV from summary.json files
# ------------------------------------------------------------
python scripts/training/build_master_csv.py \
  --family resmlp \
  --master-csv "$MASTER_CSV" \
  --runs \
  "resmlp_ae64_h512-256_do02_lr1e3_bs4096_v1" \
  "resmlp_ae64_h512-256_do03_lr7e4_bs4096_v2" \
  "resmlp_ae64_h512-256_do025_lr5e4_bs4096_v3" \
  "resmlp_ae64_h256-256_do02_lr1e3_bs4096_v4" \
  "resmlp_ae64_h256-256-256_do025_lr7e4_bs4096_v5" \
  "resmlp_ae64_h512-512-256_do03_lr7e4_bs4096_v6" \
  "resmlp_ae64_h768-384_do03_lr7e4_bs4096_v7" \
  "resmlp_ae64_h512-256_do025_lr7e4_bs2048_v8" \
  "resmlp_ae64_h512-256_do02_lr8e4_bs8192_v9" \
  "resmlp_ae64_h384-384-192_do03_lr5e4_bs4096_v10"

if false; then
python - <<'PY'
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

runs = [
    "resmlp_ae64_h512-256_do02_lr1e3_bs4096_v1",
    "resmlp_ae64_h512-256_do03_lr7e4_bs4096_v2",
    "resmlp_ae64_h512-256_do025_lr5e4_bs4096_v3",
    "resmlp_ae64_h256-256_do02_lr1e3_bs4096_v4",
    "resmlp_ae64_h256-256-256_do025_lr7e4_bs4096_v5",
    "resmlp_ae64_h512-512-256_do03_lr7e4_bs4096_v6",
    "resmlp_ae64_h768-384_do03_lr7e4_bs4096_v7",
    "resmlp_ae64_h512-256_do025_lr7e4_bs2048_v8",
    "resmlp_ae64_h512-256_do02_lr8e4_bs8192_v9",
    "resmlp_ae64_h384-384-192_do03_lr5e4_bs4096_v10",
]

master_csv = Path("runs/resmlp_ae64_master_runs.csv")
master_csv.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "run_dir",
    "timestamp_jst",
    "data_npz",
    "model",
    "model_type",
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

    hidden_dims = s.get("hidden_dims", "")
    if isinstance(hidden_dims, list):
        hidden_dims = "-".join(str(x) for x in hidden_dims)

    rows.append({
        "run_dir": str(run_dir),
        "timestamp_jst": s.get("created_at_jst", ""),
        "data_npz": s.get("data", ""),
        "model": "ResMLP",
        "model_type": s.get("model_type", ""),
        "seed": s.get("seed", ""),
        "input_dim": s.get("input_dim", ""),
        "num_classes": s.get("num_classes", ""),
        "hidden_dims": hidden_dims,
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
fi

echo
echo "Done."
echo "Master CSV: $MASTER_CSV"
