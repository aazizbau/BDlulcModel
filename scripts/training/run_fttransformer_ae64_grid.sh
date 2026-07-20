#!/usr/bin/env bash
# ==============================================================================
# Reproduction and AOI adaptation
# ==============================================================================
# Purpose: Run fttransformer ae64 grid.
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
#   bash scripts/training/run_fttransformer_ae64_grid.sh
#
# Outputs and logs are controlled by the variables below. Use a new output/log
# location for a new AOI, retain the run manifest, and inspect failures before
# resuming. Existing usage notes and worked commands below remain authoritative.
# ==============================================================================
set -euo pipefail

# ============================================================
# FT-Transformer grid search runner for AE64 classification
# ============================================================
#
# Runs 10 experiments and writes:
#   - per-run logs:
#       logs/fttransformer_ae64_grid/<run_name>.log
#   - master comparison CSV:
#       runs/fttransformer_ae64_master_runs.csv
#
# Usage:
#   chmod +x scripts/training/run_fttransformer_ae64_grid.sh
#   ./scripts/training/run_fttransformer_ae64_grid.sh
#
# Notes:
# - Assumes you run from repo root.
# - Assumes your venv is already activated.
# - Skips a run if summary.json already exists in that run dir.
# - Rebuilds the master CSV from the known run list each time.
# - Designed around the observed baseline:
#     v1 best val macro F1 = 0.7191 at epoch 14
#     test macro F1 = 0.7129
# - Search logic focuses on:
#     * slightly smaller / larger token width
#     * 2 vs 3 transformer blocks
#     * lighter / stronger dropout
#     * lower learning rate around the promising 1e-3 baseline
#     * memory-safe physical batch sizes with AMP and grad accumulation
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA="data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz"
TRAIN_SCRIPT="scripts/training/train_fttransformer_ae64_from_npz.py"
LOG_DIR="logs/fttransformer_ae64_grid"
MASTER_CSV="runs/fttransformer_ae64_master_runs.csv"

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
# - stable and memory-safe with dt128, blk2, bs512, acc4, amp
# - best val macro F1 improved until epoch 14, then plateaued
# - scheduler reductions helped modestly, but gains after epoch 14 were limited
#
# Search logic:
# - keep strong baseline as v1
# - try smaller dt96 for regularization / speed
# - try dt160 for more capacity while keeping memory safe
# - compare 2 vs 3 blocks
# - test slightly lower lr (7e-4, 5e-4)
# - test lighter vs stronger dropout
# - keep AMP on all runs
# - use grad accumulation to preserve effective batch while controlling memory
# ------------------------------------------------------------

run_one "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr1e3_bs512_acc4_amp_v1" \
  --d-token 128 \
  --n-blocks 2 \
  --n-heads 8 \
  --attention-dropout 0.1 \
  --ff-dropout 0.1 \
  --residual-dropout 0.0 \
  --batch-size 512 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt96_blk2_head8_attndo01_ffdo01_lr1e3_bs512_acc4_amp_v2" \
  --d-token 96 \
  --n-blocks 2 \
  --n-heads 8 \
  --attention-dropout 0.1 \
  --ff-dropout 0.1 \
  --residual-dropout 0.0 \
  --batch-size 512 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt160_blk2_head8_attndo01_ffdo01_lr1e3_bs384_acc4_amp_v3" \
  --d-token 160 \
  --n-blocks 2 \
  --n-heads 8 \
  --attention-dropout 0.1 \
  --ff-dropout 0.1 \
  --residual-dropout 0.0 \
  --batch-size 384 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt128_blk3_head8_attndo01_ffdo01_lr1e3_bs256_acc4_amp_v4" \
  --d-token 128 \
  --n-blocks 3 \
  --n-heads 8 \
  --attention-dropout 0.1 \
  --ff-dropout 0.1 \
  --residual-dropout 0.0 \
  --batch-size 256 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt128_blk2_head8_attndo005_ffdo005_lr1e3_bs512_acc4_amp_v5" \
  --d-token 128 \
  --n-blocks 2 \
  --n-heads 8 \
  --attention-dropout 0.05 \
  --ff-dropout 0.05 \
  --residual-dropout 0.0 \
  --batch-size 512 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt128_blk2_head8_attndo015_ffdo015_lr1e3_bs512_acc4_amp_v6" \
  --d-token 128 \
  --n-blocks 2 \
  --n-heads 8 \
  --attention-dropout 0.15 \
  --ff-dropout 0.15 \
  --residual-dropout 0.0 \
  --batch-size 512 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr7e4_bs512_acc4_amp_v7" \
  --d-token 128 \
  --n-blocks 2 \
  --n-heads 8 \
  --attention-dropout 0.1 \
  --ff-dropout 0.1 \
  --residual-dropout 0.0 \
  --batch-size 512 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr5e4_bs512_acc4_amp_v8" \
  --d-token 128 \
  --n-blocks 2 \
  --n-heads 8 \
  --attention-dropout 0.1 \
  --ff-dropout 0.1 \
  --residual-dropout 0.0 \
  --batch-size 512 \
  --grad-accum-steps 4 \
  --epochs 100 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt160_blk3_head8_attndo01_ffdo01_lr7e4_bs256_acc4_amp_v9" \
  --d-token 160 \
  --n-blocks 3 \
  --n-heads 8 \
  --attention-dropout 0.1 \
  --ff-dropout 0.1 \
  --residual-dropout 0.0 \
  --batch-size 256 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

run_one "ftt_ae64_dt96_blk3_head8_attndo015_ffdo01_lr7e4_bs256_acc4_amp_v10" \
  --d-token 96 \
  --n-blocks 3 \
  --n-heads 8 \
  --attention-dropout 0.15 \
  --ff-dropout 0.1 \
  --residual-dropout 0.0 \
  --batch-size 256 \
  --grad-accum-steps 4 \
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
  --amp \
  --device cuda \
  --seed 42

# ------------------------------------------------------------
# Rebuild master CSV from summary.json files
# ------------------------------------------------------------
python scripts/training/build_master_csv.py \
  --family fttransformer \
  --master-csv "$MASTER_CSV" \
  --runs \
  "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr1e3_bs512_acc4_amp_v1" \
  "ftt_ae64_dt96_blk2_head8_attndo01_ffdo01_lr1e3_bs512_acc4_amp_v2" \
  "ftt_ae64_dt160_blk2_head8_attndo01_ffdo01_lr1e3_bs384_acc4_amp_v3" \
  "ftt_ae64_dt128_blk3_head8_attndo01_ffdo01_lr1e3_bs256_acc4_amp_v4" \
  "ftt_ae64_dt128_blk2_head8_attndo005_ffdo005_lr1e3_bs512_acc4_amp_v5" \
  "ftt_ae64_dt128_blk2_head8_attndo015_ffdo015_lr1e3_bs512_acc4_amp_v6" \
  "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr7e4_bs512_acc4_amp_v7" \
  "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr5e4_bs512_acc4_amp_v8" \
  "ftt_ae64_dt160_blk3_head8_attndo01_ffdo01_lr7e4_bs256_acc4_amp_v9" \
  "ftt_ae64_dt96_blk3_head8_attndo015_ffdo01_lr7e4_bs256_acc4_amp_v10"

if false; then
python - <<'PY'
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

runs = [
    "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr1e3_bs512_acc4_amp_v1",
    "ftt_ae64_dt96_blk2_head8_attndo01_ffdo01_lr1e3_bs512_acc4_amp_v2",
    "ftt_ae64_dt160_blk2_head8_attndo01_ffdo01_lr1e3_bs384_acc4_amp_v3",
    "ftt_ae64_dt128_blk3_head8_attndo01_ffdo01_lr1e3_bs256_acc4_amp_v4",
    "ftt_ae64_dt128_blk2_head8_attndo005_ffdo005_lr1e3_bs512_acc4_amp_v5",
    "ftt_ae64_dt128_blk2_head8_attndo015_ffdo015_lr1e3_bs512_acc4_amp_v6",
    "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr7e4_bs512_acc4_amp_v7",
    "ftt_ae64_dt128_blk2_head8_attndo01_ffdo01_lr5e4_bs512_acc4_amp_v8",
    "ftt_ae64_dt160_blk3_head8_attndo01_ffdo01_lr7e4_bs256_acc4_amp_v9",
    "ftt_ae64_dt96_blk3_head8_attndo015_ffdo01_lr7e4_bs256_acc4_amp_v10",
]

master_csv = Path("runs/fttransformer_ae64_master_runs.csv")
master_csv.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "run_dir",
    "timestamp_jst",
    "data_npz",
    "model",
    "seed",
    "input_dim",
    "num_classes",
    "d_token",
    "n_blocks",
    "n_heads",
    "attention_dropout",
    "ff_dropout",
    "residual_dropout",
    "ff_multiplier",
    "batch_size",
    "grad_accum_steps",
    "effective_batch_size",
    "amp_enabled",
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
        "model": "FT-Transformer",
        "seed": s.get("seed", ""),
        "input_dim": s.get("input_dim", ""),
        "num_classes": s.get("num_classes", ""),
        "d_token": s.get("d_token", ""),
        "n_blocks": s.get("n_blocks", ""),
        "n_heads": s.get("n_heads", ""),
        "attention_dropout": s.get("attention_dropout", ""),
        "ff_dropout": s.get("ff_dropout", ""),
        "residual_dropout": s.get("residual_dropout", ""),
        "ff_multiplier": s.get("ff_multiplier", ""),
        "batch_size": s.get("batch_size", ""),
        "grad_accum_steps": s.get("grad_accum_steps", ""),
        "effective_batch_size": s.get("effective_batch_size", ""),
        "amp_enabled": s.get("amp_enabled", ""),
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
