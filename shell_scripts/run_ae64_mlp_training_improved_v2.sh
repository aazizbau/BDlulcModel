#!/usr/bin/env bash
# ------------------------------------------------------------
# Run improved AE64 MLP training experiments (v2)
# - focal + prior-logit adjustment (tau)
# - optional no-bn
# - optional grid-search
# - optional focal-only run
#
# Usage:
#   chmod +x run_ae64_mlp_training_improved_v2.sh
#   bash ./run_ae64_mlp_training_improved_v2.sh
# ------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ----------------------------
# USER CONFIG
# ----------------------------
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"   # or: /path/to/.venv/bin/python
SCRIPT="${PROJECT_ROOT}/scripts/training/train_mlp_ae64_v2.py"
DATA="${PROJECT_ROOT}/data/processed/training/ae64_samples_4upazila_2023.npz"
OUT_BASE="${PROJECT_ROOT}/runs"

EPOCHS_MAIN=60
EPOCHS_GRID=40
BATCH=4096

# Main hyperparams (your V2 command)
LR_MAIN="7e-4"
WD_MAIN="1e-3"
DROPOUT_MAIN="0.2"

# For focal-only run (you can adjust epochs/batch if you want)
LR_FOCAL="7e-4"
WD_FOCAL="1e-3"
DROPOUT_FOCAL="0.2"
EPOCHS_FOCAL=60

# ----------------------------
# LOGGING
# ----------------------------
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${PROJECT_ROOT}/outputs/train_logs_mlp_ae64_improved_${TS}"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date +"%F %T")] $*"; }

run_one () {
  local run_name="$1"; shift
  local log_file="${LOG_DIR}/${run_name}.log"

  log "RUN: ${run_name}"
  log "  -> logging to: ${log_file}"
  log "  -> cmd: ${PYTHON_BIN} ${SCRIPT} --data ${DATA} --run-name ${run_name} $*"

  ${PYTHON_BIN} "${SCRIPT}" \
    --data "${DATA}" \
    --run-name "${run_name}" \
    "$@" 2>&1 | tee "${log_file}"

  log "DONE: ${run_name}"
  echo
}

# ----------------------------
# SAFETY CHECKS
# ----------------------------
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: training script not found: ${SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${DATA}" ]]; then
  echo "ERROR: dataset not found: ${DATA}" >&2
  exit 1
fi

# ----------------------------
# 1) Main run (V2)
# ----------------------------
run_one "ae64_mlp_improved_focal_tau1_V2" \
  --out-dir "${OUT_BASE}" \
  --epochs "${EPOCHS_MAIN}" --batch-size "${BATCH}" \
  --lr "${LR_MAIN}" --weight-decay "${WD_MAIN}" --dropout "${DROPOUT_MAIN}" \
  --tau 1.0 \
  --amp

# ----------------------------
# 2) tau sweep (V3, V4)
# ----------------------------
run_one "tau05_V3" \
  --out-dir "${OUT_BASE}" \
  --epochs "${EPOCHS_MAIN}" --batch-size "${BATCH}" \
  --lr "${LR_MAIN}" --weight-decay "${WD_MAIN}" --dropout "${DROPOUT_MAIN}" \
  --tau 0.5 \
  --amp

run_one "tau15_V4" \
  --out-dir "${OUT_BASE}" \
  --epochs "${EPOCHS_MAIN}" --batch-size "${BATCH}" \
  --lr "${LR_MAIN}" --weight-decay "${WD_MAIN}" --dropout "${DROPOUT_MAIN}" \
  --tau 1.5 \
  --amp

# ----------------------------
# 3) no-bn run (V5)
# ----------------------------
run_one "nobn_V5" \
  --out-dir "${OUT_BASE}" \
  --epochs "${EPOCHS_MAIN}" --batch-size "${BATCH}" \
  --lr "${LR_MAIN}" --weight-decay "${WD_MAIN}" --dropout "${DROPOUT_MAIN}" \
  --tau 1.0 \
  --no-bn \
  --amp

# ----------------------------
# 4) grid-search run (V6)
# ----------------------------
run_one "ae64_grid_v1_V6" \
  --out-dir "${OUT_BASE}" \
  --grid-search \
  --epochs "${EPOCHS_GRID}" --batch-size "${BATCH}" \
  --amp

# ----------------------------
# 5) focal-only run (V7)
# ----------------------------
run_one "ae64_focal_V7" \
  --out-dir "${OUT_BASE}" \
  --epochs "${EPOCHS_FOCAL}" --batch-size "${BATCH}" \
  --lr "${LR_FOCAL}" --weight-decay "${WD_FOCAL}" --dropout "${DROPOUT_FOCAL}" \
  --loss focal \
  --amp

log "ALL DONE."
log "Logs saved under: ${LOG_DIR}"
