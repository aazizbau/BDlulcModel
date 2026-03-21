#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1D-CNN grid search runner for AE64 classification
# ============================================================
#
# Runs 10 experiments and writes:
#   - per-run logs:
#       logs/cnn1d_ae64_grid/<run_name>.log
#   - master comparison CSV:
#       runs/cnn1d_ae64_master_runs.csv
#
# Usage:
#   chmod +x scripts/training/run_cnn1d_ae64_grid.sh
#   ./scripts/training/run_cnn1d_ae64_grid.sh
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
TRAIN_SCRIPT="scripts/training/train_cnn1d_ae64_from_npz.py"
LOG_DIR="logs/cnn1d_ae64_grid"
MASTER_CSV="runs/cnn1d_ae64_master_runs.csv"

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
# - strong steady improvement up to later epochs
# - scheduler helped, with best validation macro F1 around epoch 85
# - final train/val gap is modest, so search should explore both:
#     (a) slightly larger capacity
#     (b) slightly stronger regularization
#     (c) learning-rate / batch-size stability
#
# Search logic:
# - vary width: narrower / baseline / wider
# - vary receptive field using kernel patterns
# - vary classifier head size
# - vary dropout around 0.2
# - vary learning rate around 1e-3
# - vary weight decay mildly
# - vary batch size to test optimization stability
# ------------------------------------------------------------

run_one "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs4096_v1" \
  --channels 32 64 128 \
  --kernels 5 3 3 \
  --head-dim 128 \
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

run_one "cnn1d_ae64_c32-64-128_k7-5-3_h128_do02_lr1e3_bs4096_v2" \
  --channels 32 64 128 \
  --kernels 7 5 3 \
  --head-dim 128 \
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

run_one "cnn1d_ae64_c32-64-128_k5-3-3_h256_do02_lr1e3_bs4096_v3" \
  --channels 32 64 128 \
  --kernels 5 3 3 \
  --head-dim 256 \
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

run_one "cnn1d_ae64_c64-128-128_k5-3-3_h128_do02_lr1e3_bs4096_v4" \
  --channels 64 128 128 \
  --kernels 5 3 3 \
  --head-dim 128 \
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

run_one "cnn1d_ae64_c32-64-128_k5-3-3_h128_do03_lr1e3_bs4096_v5" \
  --channels 32 64 128 \
  --kernels 5 3 3 \
  --head-dim 128 \
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

run_one "cnn1d_ae64_c32-64-128_k5-3-3_h128_do015_lr1e3_bs4096_v6" \
  --channels 32 64 128 \
  --kernels 5 3 3 \
  --head-dim 128 \
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

run_one "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr7e4_bs4096_v7" \
  --channels 32 64 128 \
  --kernels 5 3 3 \
  --head-dim 128 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 7e-4 \
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

run_one "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr15e4_bs4096_v8" \
  --channels 32 64 128 \
  --kernels 5 3 3 \
  --head-dim 128 \
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

run_one "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs8192_v9" \
  --channels 32 64 128 \
  --kernels 5 3 3 \
  --head-dim 128 \
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

run_one "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs4096_wd5e4_v10" \
  --channels 32 64 128 \
  --kernels 5 3 3 \
  --head-dim 128 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

# ------------------------------------------------------------
# Rebuild master CSV from summary.json files
# ------------------------------------------------------------
python scripts/training/build_master_csv.py \
  --family cnn1d \
  --master-csv "$MASTER_CSV" \
  --runs \
  "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs4096_v1" \
  "cnn1d_ae64_c32-64-128_k7-5-3_h128_do02_lr1e3_bs4096_v2" \
  "cnn1d_ae64_c32-64-128_k5-3-3_h256_do02_lr1e3_bs4096_v3" \
  "cnn1d_ae64_c64-128-128_k5-3-3_h128_do02_lr1e3_bs4096_v4" \
  "cnn1d_ae64_c32-64-128_k5-3-3_h128_do03_lr1e3_bs4096_v5" \
  "cnn1d_ae64_c32-64-128_k5-3-3_h128_do015_lr1e3_bs4096_v6" \
  "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr7e4_bs4096_v7" \
  "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr15e4_bs4096_v8" \
  "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs8192_v9" \
  "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs4096_wd5e4_v10"

if false; then
python - <<'PY'
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

runs = [
    "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs4096_v1",
    "cnn1d_ae64_c32-64-128_k7-5-3_h128_do02_lr1e3_bs4096_v2",
    "cnn1d_ae64_c32-64-128_k5-3-3_h256_do02_lr1e3_bs4096_v3",
    "cnn1d_ae64_c64-128-128_k5-3-3_h128_do02_lr1e3_bs4096_v4",
    "cnn1d_ae64_c32-64-128_k5-3-3_h128_do03_lr1e3_bs4096_v5",
    "cnn1d_ae64_c32-64-128_k5-3-3_h128_do015_lr1e3_bs4096_v6",
    "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr7e4_bs4096_v7",
    "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr15e4_bs4096_v8",
    "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs8192_v9",
    "cnn1d_ae64_c32-64-128_k5-3-3_h128_do02_lr1e3_bs4096_wd5e4_v10",
]

master_csv = Path("runs/cnn1d_ae64_master_runs.csv")
master_csv.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "run_dir",
    "timestamp_jst",
    "data_npz",
    "model",
    "seed",
    "input_dim",
    "num_classes",
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

    rows.append({
        "run_dir": str(run_dir),
        "timestamp_jst": s.get("created_at_jst", ""),
        "data_npz": s.get("data", ""),
        "model": "CNN1D",
        "seed": s.get("seed", ""),
        "input_dim": s.get("input_dim", ""),
        "num_classes": s.get("num_classes", ""),
        "channels": "|".join(map(str, s.get("channels", []))) if isinstance(s.get("channels", []), list) else s.get("channels", ""),
        "kernels": "|".join(map(str, s.get("kernels", []))) if isinstance(s.get("kernels", []), list) else s.get("kernels", ""),
        "head_dim": s.get("head_dim", ""),
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
