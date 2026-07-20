#!/usr/bin/env bash
# ==============================================================================
# Reproduction and AOI adaptation
# ==============================================================================
# Purpose: Run calculate s2 10indices multiyear.
# Workflow role: Calculate Sentinel-2 spectral indices from aligned reflectance bands.
#
# Prerequisites:
#   1. Run from the repository root with the project environment activated.
#   2. Install requirements.txt and any system GDAL/Earth Engine dependencies.
#   3. Verify every input path and available disk/GPU resources before starting.
#
# AOI adaptation:
#   Replace band paths with aligned reflectance rasters for the target AOI and keep nodata masks, grid geometry, and scale factors consistent.
#   Keep CRS, resolution, nodata, feature order, class IDs, and split metadata
#   consistent across all scripts invoked by this runner.
#
# Reproducible example:
#   bash scripts/s2_indices/run_calculate_s2_10indices_multiyear.sh
#
# Outputs and logs are controlled by the variables below. Use a new output/log
# location for a new AOI, retain the run manifest, and inspect failures before
# resuming. Existing usage notes and worked commands below remain authoritative.
# ==============================================================================
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
LOG_DIR="${PROJECT_ROOT}/logs/calculate_s2_10indices"
MASTER_LOG="${LOG_DIR}/run_calculate_s2_10indices_multiyear.log"
YEARS=(2017 2023 2024)

ts() {
  date +"[%Y-%m-%dT%H:%M:%S%z]"
}

log() {
  echo "$(ts) $*" | tee -a "${MASTER_LOG}"
}

run_one() {
  local year="$1"
  local index_name="$2"
  local script_path="$3"
  local log_file="${LOG_DIR}/${year}_${index_name}.log"

  log "START year=${year} index=${index_name} script=${script_path}"
  {
    echo "$(ts) START year=${year} index=${index_name}"
    echo "$(ts) CMD: ${PYTHON_BIN} ${script_path} --year ${year}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/${script_path}" --year "${year}"
    echo "$(ts) END year=${year} index=${index_name} status=success"
  } 2>&1 | tee "${log_file}"

  log "DONE  year=${year} index=${index_name} log=${log_file}"
}

main() {
  mkdir -p "${LOG_DIR}"
  : > "${MASTER_LOG}"

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: PYTHON_BIN not found or not executable: ${PYTHON_BIN}" | tee -a "${MASTER_LOG}" >&2
    exit 1
  fi

  log "Project root: ${PROJECT_ROOT}"
  log "Python bin  : ${PYTHON_BIN}"
  log "Log dir     : ${LOG_DIR}"
  log "Years       : ${YEARS[*]}"

  for year in "${YEARS[@]}"; do
    run_one "${year}" "ndvi"    "scripts/s2_indices/make_ndvi_image.py"
    run_one "${year}" "evi"     "scripts/s2_indices/make_evi_image.py"
    run_one "${year}" "msavi"   "scripts/s2_indices/make_msavi_image.py"
    run_one "${year}" "ndmi"    "scripts/s2_indices/make_ndmi_image.py"
    run_one "${year}" "ndwi"    "scripts/s2_indices/make_ndwi_image.py"
    run_one "${year}" "ndpi"    "scripts/s2_indices/make_ndpi_image.py"
    run_one "${year}" "ndbi"    "scripts/s2_indices/make_ndbi_image.py"
    run_one "${year}" "bsi"     "scripts/s2_indices/make_bsi_image.py"
    run_one "${year}" "nirv"    "scripts/s2_indices/make_nirv_image.py"
    run_one "${year}" "awei_sh" "scripts/s2_indices/make_awei_sh_image.py"
  done

  log "All index calculations completed successfully."
}

main "$@"
