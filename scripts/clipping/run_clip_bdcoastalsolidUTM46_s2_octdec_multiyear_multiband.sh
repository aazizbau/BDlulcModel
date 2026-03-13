#!/usr/bin/env bash
set -euo pipefail

# Run clipping for multiple years and bands:
#   years: 2017, 2023, 2024
#   bands: B2, B3, B4, B8, B11, B12
#
# It calls:
#   python scripts/clipping/clip_tif_toUTM46_bdsolid_coastal.py --year <YEAR> --band <BAND>
#
# Outputs:
#   - per-run logs
#   - a master summary CSV
#   - a run manifest text file
#   - a latest symlink to the newest batch log directory
#
# Example:
#   bash scripts/clipping/run_clip_bdcoastalsolidUTM46_s2_octdec_multiyear_multiband.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CLIP_SCRIPT="scripts/clipping/clip_tif_toUTM46_bdsolid_coastal.py"

LOG_ROOT="logs/clip_toUTM46_bdcoastal_solid"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${LOG_ROOT}/run_${RUN_TS}"
MASTER_LOG="${RUN_DIR}/master.log"
MANIFEST_TXT="${RUN_DIR}/manifest.txt"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
LATEST_LINK="${LOG_ROOT}/latest"

YEARS=(2017 2023 2024)
BANDS=(B2 B3 B4 B8 B11 B12)

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${MASTER_LOG}"
}

ensure_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        printf '[%s] ERROR: Required file not found: %s\n' "$(timestamp)" "${path}" | tee -a "${MASTER_LOG}" >&2
        exit 1
    fi
}

mkdir -p "${RUN_DIR}"

: > "${MASTER_LOG}"
: > "${MANIFEST_TXT}"

cat > "${SUMMARY_CSV}" <<'CSV'
run_timestamp,year,band,status,exit_code,log_file,output_tif,start_time,end_time,duration_sec
CSV

log "Starting multiyear multiband clipping run"
log "Project root : ${PROJECT_ROOT}"
log "Python bin   : ${PYTHON_BIN}"
log "Clip script  : ${CLIP_SCRIPT}"
log "Run dir      : ${RUN_DIR}"

ensure_file "${CLIP_SCRIPT}"

{
    echo "run_timestamp=${RUN_TS}"
    echo "project_root=${PROJECT_ROOT}"
    echo "python_bin=${PYTHON_BIN}"
    echo "clip_script=${CLIP_SCRIPT}"
    echo "log_root=${LOG_ROOT}"
    echo "run_dir=${RUN_DIR}"
    echo "years=${YEARS[*]}"
    echo "bands=${BANDS[*]}"
} >> "${MANIFEST_TXT}"

success_count=0
fail_count=0
total_count=0

for year in "${YEARS[@]}"; do
    for band in "${BANDS[@]}"; do
        total_count=$((total_count + 1))

        start_epoch="$(date +%s)"
        start_time="$(timestamp)"
        per_log="${RUN_DIR}/clip_${year}_${band}.log"
        output_tif="data/interim/S2_${year}_${band}_10m_utm46_bdcoastal_solid.tif"

        log "------------------------------------------------------------"
        log "Running year=${year} band=${band}"
        log "Log file    : ${per_log}"
        log "Output tif  : ${output_tif}"

        cmd=(
            "${PYTHON_BIN}" "${CLIP_SCRIPT}"
            --year "${year}"
            --band "${band}"
        )

        {
            printf '[%s] COMMAND: ' "$(timestamp)"
            printf '%q ' "${cmd[@]}"
            printf '\n'
        } | tee "${per_log}" >> "${MASTER_LOG}"

        set +e
        "${cmd[@]}" >> "${per_log}" 2>&1
        exit_code=$?
        set -e

        end_epoch="$(date +%s)"
        end_time="$(timestamp)"
        duration_sec=$((end_epoch - start_epoch))

        if [[ ${exit_code} -eq 0 ]]; then
            status="SUCCESS"
            success_count=$((success_count + 1))
            log "Finished year=${year} band=${band} status=${status} duration=${duration_sec}s"
        else
            status="FAILED"
            fail_count=$((fail_count + 1))
            log "Finished year=${year} band=${band} status=${status} exit_code=${exit_code} duration=${duration_sec}s"
        fi

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${RUN_TS}" \
            "${year}" \
            "${band}" \
            "${status}" \
            "${exit_code}" \
            "${per_log}" \
            "${output_tif}" \
            "${start_time}" \
            "${end_time}" \
            "${duration_sec}" >> "${SUMMARY_CSV}"
    done
done

rm -f "${LATEST_LINK}"
ln -s "$(basename "${RUN_DIR}")" "${LATEST_LINK}"

log "------------------------------------------------------------"
log "Batch completed"
log "Total   : ${total_count}"
log "Success : ${success_count}"
log "Failed  : ${fail_count}"
log "Summary : ${SUMMARY_CSV}"
log "Manifest: ${MANIFEST_TXT}"
log "Latest  : ${LATEST_LINK}"

if [[ ${fail_count} -gt 0 ]]; then
    exit 1
fi
