#!/bin/bash
# ==============================================================================
# Reproduction and AOI adaptation
# ==============================================================================
# Purpose: Run s2 octdec download multiyear multiband.
# Workflow role: Orchestrate a reproducible project workflow.
#
# Prerequisites:
#   1. Run from the repository root with the project environment activated.
#   2. Install requirements.txt and any system GDAL/Earth Engine dependencies.
#   3. Verify every input path and available disk/GPU resources before starting.
#
# AOI adaptation:
#   Replace project-specific paths, AOI settings, years, and output destinations before running.
#   Keep CRS, resolution, nodata, feature order, class IDs, and split metadata
#   consistent across all scripts invoked by this runner.
#
# Reproducible example:
#   bash shell_scripts/run_s2_octdec_download_multiyear_multiband.sh
#
# Outputs and logs are controlled by the variables below. Use a new output/log
# location for a new AOI, retain the run manifest, and inspect failures before
# resuming. Existing usage notes and worked commands below remain authoritative.
# ==============================================================================
# ----------------------------------------------------
# Lab-style Sentinel-2 Oct–Dec download
# Multi-year, multi-band batch runner
#
# One-time setup:
#   chmod +x run_s2_octdec_download_multiyear_multiband.sh
# ----------------------------------------------------

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
AOI="${AOI:-${PROJECT_ROOT}/configs/bd_coastal_aoi.yaml}"
PROJECT="${PROJECT:-${GEE_PROJECT_ID:-}}"

YEARS=(2017 2023 2024)
BANDS=(B02 B03 B04 B08 B11 B12)
RESUME=1

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/labstyle_download}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/labstyle_gemini_download_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN not found or not executable: ${PYTHON_BIN}" >&2
  echo "Hint: export PYTHON_BIN=/full/path/to/.venv/bin/python" >&2
  exit 1
fi

if [[ -z "${PROJECT}" ]]; then
  echo "ERROR: PROJECT is not set. Export GEE_PROJECT_ID or PROJECT." >&2
  exit 1
fi

# ----------------------------------------------------
# LOGGING FUNCTIONS
# ----------------------------------------------------
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_cmd() {
  log "START: $*"
  "$@" >> "$LOG_FILE" 2>&1
  log "SUCCESS: $*"
}

trap 'log "ERROR: Script failed at line $LINENO"; exit 1' ERR

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
log "Sentinel-2 Oct-Dec LAB-STYLE Gemini download started"
log "Years: ${YEARS[*]}"
log "Bands: ${BANDS[*]}"
log "AOI: $AOI"
log "Project: $PROJECT"

for YEAR in "${YEARS[@]}"; do
  log "----- Processing YEAR=${YEAR} -----"

  # Apply the Thesis-Safe Cloud Score threshold for 2017
  if [ "$YEAR" -eq 2017 ]; then
    CLOUD_THRESH=0.40
    log "NOTICE: Using lowered Cloud Score threshold ($CLOUD_THRESH) for 2017."
  else
    CLOUD_THRESH=0.60
  fi

  for BAND in "${BANDS[@]}"; do

    # Skip 2017 B02 since the test run already completed it successfully
    if [ "$YEAR" -eq 2017 ] && [ "$BAND" = "B02" ]; then
      log "Skipping YEAR=2017 BAND=B02 (Already completed in test)"
      continue
    fi

    log "Downloading YEAR=${YEAR} BAND=${BAND} (CS+ >= ${CLOUD_THRESH})"

    # Build the command array for clean execution
    CMD=("${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/s2_download/download_s2_octdec.py" \
      --year "$YEAR" \
      --band "$BAND" \
      --aoi "$AOI" \
      --project "$PROJECT" \
      --cloud-threshold "$CLOUD_THRESH")

    # Add resume flag if configured
    if [ "$RESUME" -eq 1 ]; then
      CMD+=(--resume)
    fi

    run_cmd "${CMD[@]}"

  done
done

log "Sentinel-2 Oct-Dec LAB-STYLE Gemini download completed successfully"
log "Log file saved at: $LOG_FILE"
