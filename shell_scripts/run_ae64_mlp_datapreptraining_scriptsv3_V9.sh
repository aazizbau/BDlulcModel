#!/usr/bin/env bash
# ------------------------------------------------------------
# Run AE64 MLP data-prep + training (v3) experiments (V9)
#
# Steps:
#  1) Rasterize labels (v3) -> assets/training_labels_v3/
#  2) Extract samples (v3)  -> data/processed/training/ae64_samples_4upazila_2023_v3.npz
#  3) Train MLP (v3):
#       - tau sweep: 1.2..1.7
#       - multi-seed: 42..46
#       - ensemble eval
#       - tensorboard logs
#
# Usage:
#   chmod +x run_ae64_mlp_datapreptraining_scriptsv3_V9.sh
#   bash ./run_ae64_mlp_datapreptraining_scriptsv3_V9.sh
#
# Optional env overrides:
#   PYTHON_BIN=/path/to/.venv/bin/python bash ./run_ae64_mlp_datapreptraining_scriptsv3_V9.sh
#   HEADLESS=0 bash ./run_ae64_mlp_datapreptraining_scriptsv3_V9.sh   # if you truly want GUI backends
# ------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ----------------------------
# USER CONFIG
# ----------------------------
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

LABEL_SCRIPT="${PROJECT_ROOT}/scripts/labels/make_training_label_v3.py"
EXTRACT_SCRIPT="${PROJECT_ROOT}/scripts/training/extract_ae_samples_v3.py"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/training/train_mlp_ae64_v3.py"

LABEL_OUT_DIR="${PROJECT_ROOT}/assets/training_labels_v3"

AE_MOSAIC="${PROJECT_ROOT}/data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif"
SAMPLES_OUT="${PROJECT_ROOT}/data/processed/training/ae64_samples_4upazila_2023_v3.npz"

RUNS_OUT_DIR="${PROJECT_ROOT}/runs"

EPOCHS=60
BATCH=4096
LR="3e-4"
WD="1e-3"
DROPOUT="0.3"

TAU_LIST="1.2,1.3,1.4,1.5,1.6,1.7"
SEEDS="42,43,44,45,46"

# Extraction knobs
MAX_PER_CLASS_PER_UPAZILA=150000
VAL_FRAC=0.2
MIN_VAL_PER_CLASS=10000
MIN_VAL_FRAC_PER_CLASS=0.02
BLOCK_SIZE_M=1000
ERODE_PX=1

# ----------------------------
# HEADLESS SAFETY (prevents Tkinter/Tcl crashes)
# ----------------------------
HEADLESS="${HEADLESS:-1}"
if [[ "${HEADLESS}" == "1" ]]; then
  export MPLBACKEND="${MPLBACKEND:-Agg}"     # <- critical: avoid TkAgg
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
  export DISPLAY="${DISPLAY:-}"
fi
export PYTHONUNBUFFERED=1

# ----------------------------
# LOGGING
# ----------------------------
TS="$(date +"%Y%m%d_%H%M%S")"
TODAY="$(date +"%Y%m%d")"

LOG_DIR="${PROJECT_ROOT}/outputs/train_logs_mlp_ae64_scriptsv3_V9_${TS}"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date +"%F %T")] $*"; }

run_cmd () {
  local name="$1"; shift
  local log_file="${LOG_DIR}/${name}.log"

  log "RUN: ${name}"
  log "  -> logging to: ${log_file}"
  log "  -> cmd: $*"

  # Use a subshell to keep pipefail behavior and capture correct exit status
  set +e
  ( "$@" ) 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    log "FAILED: ${name} (exit code ${rc})"
    log "See log: ${log_file}"
    exit $rc
  fi

  log "DONE: ${name}"
  echo
}

# ----------------------------
# SAFETY CHECKS
# ----------------------------
for f in "${LABEL_SCRIPT}" "${EXTRACT_SCRIPT}" "${TRAIN_SCRIPT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: script not found: ${f}" >&2
    exit 1
  fi
done

if [[ ! -f "${AE_MOSAIC}" ]]; then
  echo "ERROR: AE mosaic not found: ${AE_MOSAIC}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not found or not executable: ${PYTHON_BIN}" >&2
  echo "Hint: export PYTHON_BIN=/full/path/to/.venv/bin/python" >&2
  exit 1
fi

mkdir -p "${LABEL_OUT_DIR}"
mkdir -p "$(dirname "${SAMPLES_OUT}")"
mkdir -p "${RUNS_OUT_DIR}"

# ----------------------------
# 1) LABEL RASTERIZATION (v3)
# ----------------------------
run_cmd "labels_manpura_v3" \
  "${PYTHON_BIN}" "${LABEL_SCRIPT}" --upazila manpura --out-dir "${LABEL_OUT_DIR}"

run_cmd "labels_betagi_v3" \
  "${PYTHON_BIN}" "${LABEL_SCRIPT}" --upazila betagi --out-dir "${LABEL_OUT_DIR}"

run_cmd "labels_amtali_v3" \
  "${PYTHON_BIN}" "${LABEL_SCRIPT}" --upazila amtali --out-dir "${LABEL_OUT_DIR}"

run_cmd "labels_bamna_v3" \
  "${PYTHON_BIN}" "${LABEL_SCRIPT}" --upazila bamna --out-dir "${LABEL_OUT_DIR}"

# ----------------------------
# 2) SAMPLE EXTRACTION (v3)
# ----------------------------
run_cmd "extract_ae64_samples_v3" \
  "${PYTHON_BIN}" "${EXTRACT_SCRIPT}" \
    --ae "${AE_MOSAIC}" \
    --labels-dir "${LABEL_OUT_DIR}" \
    --output "${SAMPLES_OUT}" \
    --max-per-class-per-upazila "${MAX_PER_CLASS_PER_UPAZILA}" \
    --val-frac "${VAL_FRAC}" \
    --min-val-per-class "${MIN_VAL_PER_CLASS}" \
    --min-val-frac-per-class "${MIN_VAL_FRAC_PER_CLASS}" \
    --block-size-m "${BLOCK_SIZE_M}" \
    --erode-px "${ERODE_PX}" \
    --min-nonzero-bands 8 \
    --chunk 1024

if [[ ! -f "${SAMPLES_OUT}" ]]; then
  echo "ERROR: sample NPZ was not created: ${SAMPLES_OUT}" >&2
  exit 1
fi

# ----------------------------
# 3) TRAINING (v3) - tau sweep + multi-seed + ensemble eval
# ----------------------------
RUN_NAME="ae64_scriptsv3_best_sweep_${TODAY}_V9"

run_cmd "${RUN_NAME}" \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data "${SAMPLES_OUT}" \
    --run-name "${RUN_NAME}" \
    --out-dir "${RUNS_OUT_DIR}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH}" \
    --lr "${LR}" --weight-decay "${WD}" --dropout "${DROPOUT}" \
    --tau-list "${TAU_LIST}" \
    --seeds "${SEEDS}" \
    --amp --ensemble-eval --tensorboard

log "ALL DONE."
log "Logs saved under: ${LOG_DIR}"
log "Run directory: ${RUNS_OUT_DIR}/${RUN_NAME}"
