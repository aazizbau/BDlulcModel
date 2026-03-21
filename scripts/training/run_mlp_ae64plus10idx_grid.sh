#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MLP grid search runner for AE64 + 10 indices classification
# ============================================================
#
# Runs 10 experiments and writes:
#   - per-run logs:
#       logs/mlp_ae64plus10idx_grid/<run_name>.log
#   - master comparison CSV:
#       runs/mlp_ae64plus10idx_master_runs.csv
#
# Usage:
#   chmod +x scripts/training/run_mlp_ae64plus10idx_grid.sh
#   ./scripts/training/run_mlp_ae64plus10idx_grid.sh
#
# Notes:
# - Assumes you are running from repo root.
# - Assumes your venv is already activated.
# - Skips a run if summary.json already exists in that run dir.
# - Rebuilds the master CSV from the known run list each time.
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA="data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz"
SCRIPT="scripts/training/train_mlp_ae64_plus10indices_from_npz.py"

LOG_DIR="logs/mlp_ae64plus10idx_grid"
MASTER_CSV="runs/mlp_ae64plus10idx_master_runs.csv"

mkdir -p "$LOG_DIR"
mkdir -p runs

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  echo "[$(timestamp)] $*"
}

run_one() {
  local run_name="$1"
  shift

  local outdir="runs/${run_name}"
  local summary="${outdir}/summary.json"
  local logfile="${LOG_DIR}/${run_name}.log"

  if [[ -f "$summary" ]]; then
    log "SKIP  ${run_name} (summary exists)"
    return 0
  fi

  log "START ${run_name}"
  {
    echo "[$(timestamp)] START ${run_name}"
    echo "[$(timestamp)] CMD: python ${SCRIPT} --data ${DATA} --outdir ${outdir} $*"
    python "$SCRIPT" \
      --data "$DATA" \
      --outdir "$outdir" \
      "$@"
    echo "[$(timestamp)] DONE ${run_name}"
  } 2>&1 | tee "$logfile"
}

# ------------------------------------------------------------
# 10 runs total (v1 ... v10)
# ------------------------------------------------------------
# v1  : baseline from current best-known setup
# v2  : lower dropout
# v3  : higher dropout
# v4  : lower learning rate
# v5  : lower weight decay
# v6  : no label smoothing
# v7  : lighter label smoothing
# v8  : smaller batch size
# v9  : wider network
# v10 : deeper + wider network
# ------------------------------------------------------------

run_one "mlp_ae64plus10idx_h512-256_do02_lr1e3_bs4096_v1" \
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

run_one "mlp_ae64plus10idx_h512-256_do01_lr1e3_bs4096_v2" \
  --hidden 512 256 \
  --dropout 0.1 \
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

run_one "mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3" \
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

run_one "mlp_ae64plus10idx_h512-256_do02_lr5e4_bs4096_v4" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
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
  --device cuda \
  --seed 42

run_one "mlp_ae64plus10idx_h512-256_do02_lr1e3_wd5e5_bs4096_v5" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 5e-5 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64plus10idx_h512-256_do02_lr1e3_ls000_bs4096_v6" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.0 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64plus10idx_h512-256_do02_lr1e3_ls002_bs4096_v7" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.02 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

run_one "mlp_ae64plus10idx_h512-256_do02_lr1e3_bs2048_v8" \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 2048 \
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

run_one "mlp_ae64plus10idx_h1024-512_do02_lr1e3_bs4096_v9" \
  --hidden 1024 512 \
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

run_one "mlp_ae64plus10idx_h1024-512-256_do02_lr1e3_bs4096_v10" \
  --hidden 1024 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 120 \
  --lr 1e-3 \
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

# ------------------------------------------------------------
# Rebuild master CSV from known run list
# ------------------------------------------------------------
log "Rebuilding master CSV: ${MASTER_CSV}"

python scripts/training/build_master_csv.py \
  --family mlp \
  --master-csv "$MASTER_CSV" \
  --runs \
  "mlp_ae64plus10idx_h512-256_do02_lr1e3_bs4096_v1" \
  "mlp_ae64plus10idx_h512-256_do01_lr1e3_bs4096_v2" \
  "mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3" \
  "mlp_ae64plus10idx_h512-256_do02_lr5e4_bs4096_v4" \
  "mlp_ae64plus10idx_h512-256_do02_lr1e3_wd5e5_bs4096_v5" \
  "mlp_ae64plus10idx_h512-256_do02_lr1e3_ls000_bs4096_v6" \
  "mlp_ae64plus10idx_h512-256_do02_lr1e3_ls002_bs4096_v7" \
  "mlp_ae64plus10idx_h512-256_do02_lr1e3_bs2048_v8" \
  "mlp_ae64plus10idx_h1024-512_do02_lr1e3_bs4096_v9" \
  "mlp_ae64plus10idx_h1024-512-256_do02_lr1e3_bs4096_v10"

if false; then
python - <<'PY'
import csv
import json
from pathlib import Path

runs = [
    "mlp_ae64plus10idx_h512-256_do02_lr1e3_bs4096_v1",
    "mlp_ae64plus10idx_h512-256_do01_lr1e3_bs4096_v2",
    "mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3",
    "mlp_ae64plus10idx_h512-256_do02_lr5e4_bs4096_v4",
    "mlp_ae64plus10idx_h512-256_do02_lr1e3_wd5e5_bs4096_v5",
    "mlp_ae64plus10idx_h512-256_do02_lr1e3_ls000_bs4096_v6",
    "mlp_ae64plus10idx_h512-256_do02_lr1e3_ls002_bs4096_v7",
    "mlp_ae64plus10idx_h512-256_do02_lr1e3_bs2048_v8",
    "mlp_ae64plus10idx_h1024-512_do02_lr1e3_bs4096_v9",
    "mlp_ae64plus10idx_h1024-512-256_do02_lr1e3_bs4096_v10",
]

out_csv = Path("runs/mlp_ae64plus10idx_master_runs.csv")
rows = []

def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

for name in runs:
    run_dir = Path("runs") / name
    summary = load_json(run_dir / "summary.json")
    if not summary:
        continue

    row = {
        "run_dir": str(run_dir),
        "timestamp_jst": summary.get("created_at_jst"),
        "data_npz": summary.get("data"),
        "model": "MLPClassifier",
        "seed": summary.get("seed"),
        "input_dim": summary.get("input_dim"),
        "num_classes": summary.get("num_classes"),
        "hidden_dims": json.dumps(summary.get("hidden_dims")),
        "dropout": summary.get("dropout"),
        "batch_size": summary.get("batch_size"),
        "epochs_requested": summary.get("epochs_requested"),
        "epochs_completed": summary.get("epochs_completed"),
        "lr": summary.get("lr"),
        "weight_decay": summary.get("weight_decay"),
        "label_smoothing": summary.get("label_smoothing"),
        "scheduler": summary.get("scheduler"),
        "scheduler_factor": summary.get("scheduler_factor"),
        "scheduler_patience": summary.get("scheduler_patience"),
        "train_samples": summary.get("train_samples"),
        "val_samples": summary.get("val_samples"),
        "test_samples": summary.get("test_samples"),
        "best_epoch": summary.get("best_epoch"),
        "best_val_macro_f1": summary.get("best_val_macro_f1"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_acc": summary.get("best_val_acc"),
        "best_val_balanced_acc": summary.get("best_val_balanced_acc"),
        "test_evaluated": summary.get("test_evaluated"),
        "test_loss": summary.get("test_loss"),
        "test_acc": summary.get("test_acc"),
        "test_macro_f1": summary.get("test_macro_f1"),
        "test_balanced_acc": summary.get("test_balanced_acc"),
        "total_train_seconds": summary.get("total_train_seconds"),
        "best_model_path": summary.get("best_model_path"),
        "history_csv": summary.get("history_csv"),
        "confusion_matrix_val_csv": summary.get("confusion_matrix_val_csv"),
        "confusion_matrix_test_csv": summary.get("confusion_matrix_test_csv"),
        "notes": "AE64+10indices MLP classification grid",
    }
    rows.append(row)

def safe_float_desc(x, fallback=-1e18):
    try:
        if x is None:
            return fallback
        return float(x)
    except Exception:
        return fallback

def safe_float_asc(x, fallback=1e18):
    try:
        if x is None:
            return fallback
        return float(x)
    except Exception:
        return fallback

rows.sort(
    key=lambda r: (
        -safe_float_desc(r.get("best_val_macro_f1")),
        -safe_float_desc(r.get("test_macro_f1")),
        safe_float_asc(r.get("best_val_loss")),
    )
)

out_csv.parent.mkdir(parents=True, exist_ok=True)
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
    "best_model_path",
    "history_csv",
    "confusion_matrix_val_csv",
    "confusion_matrix_test_csv",
    "notes",
]

with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        w.writerow(row)

print(f"[MASTER] wrote: {out_csv}")
if rows:
    best = rows[0]
    print(
        "[MASTER] best:"
        f" best_val_macro_f1={best['best_val_macro_f1']}"
        f" test_macro_f1={best['test_macro_f1']}"
        f" best_val_loss={best['best_val_loss']}"
        f" run_dir={best['run_dir']}"
    )
else:
    print("[MASTER] no completed runs found")
PY
fi

log "All done."
