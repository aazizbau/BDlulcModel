#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# =========================
# CONFIGURATION
# =========================
OLD_VENV_PY="${OLD_VENV_PY:-${PROJECT_ROOT}/.venv/bin/python}"
PROJECT="${PROJECT:-${GEE_PROJECT_ID:-}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/alphaearth_2023_pipeline_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

if [[ -z "${PROJECT}" ]]; then
    echo "ERROR: PROJECT is not set. Export GEE_PROJECT_ID or PROJECT." >&2
    exit 1
fi

# =========================
# LOGGING FUNCTIONS
# =========================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_cmd() {
    log "START: $*"
    "$@" >>"$LOG_FILE" 2>&1
    log "SUCCESS: $*"
}

trap 'log "ERROR: Command failed at line $LINENO. Pipeline stopped."; exit 1' ERR

# =========================
# PIPELINE
# =========================

log "Pipeline started"
log "No wait — running immediately"

# -------------------------------------------------
# 1. Check AlphaEarth tiles (using existing venv)
# -------------------------------------------------
run_cmd "$OLD_VENV_PY" "${PROJECT_ROOT}/scripts/gee/check_alphaearth_tiles.py" \
    --output "${PROJECT_ROOT}/data/raw/embeddings/bd_coastal_alphaearth_2023.tif" \
    --project "$PROJECT"

# -------------------------------------------------
# 2. Download missing AlphaEarth tiles
# -------------------------------------------------
run_cmd "$OLD_VENV_PY" "${PROJECT_ROOT}/scripts/gee/download_missing_alphaearth_tiles.py" \
    --year 2023 \
    --output "${PROJECT_ROOT}/data/raw/embeddings/bd_coastal_alphaearth_2023.tif" \
    --project "$PROJECT"

# -------------------------------------------------
# 3. Mosaic AlphaEarth tiles
# -------------------------------------------------
run_cmd "$OLD_VENV_PY" "${PROJECT_ROOT}/scripts/gee/mosaic_alphaearth_tiles_faster.py" \
    --input-base "${PROJECT_ROOT}/data/raw/embeddings/bd_coastal_alphaearth_2023.tif" \
    --output "${PROJECT_ROOT}/data/interim/bd_coastal_alphaearth_2023_mosaic.tif" \
    --gdal-cache-mb 4096

# -------------------------------------------------
# 4. Repair Python virtual environment
# -------------------------------------------------
run_cmd rm -rf "${PROJECT_ROOT}/.venv"
run_cmd python3 -m venv "${PROJECT_ROOT}/.venv"

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
VENV_PIP="${PROJECT_ROOT}/.venv/bin/pip"

# -------------------------------------------------
# 5. Install dependencies
# -------------------------------------------------
run_cmd "$VENV_PIP" install -r "${PROJECT_ROOT}/requirements.txt"

# -------------------------------------------------
# 6. Clip 2023 AlphaEarth mosaic
# -------------------------------------------------
run_cmd "$VENV_PY" "${PROJECT_ROOT}/scripts/gee/clip_alphaearth_mosaic.py" \
    --input "${PROJECT_ROOT}/data/interim/bd_coastal_alphaearth_2023_mosaic.tif" \
    --output "${PROJECT_ROOT}/data/processed/features/bd_coastal_alphaearth_2023_clipped.tif"

# -------------------------------------------------
# 7. Clip 2024 AlphaEarth mosaic
# -------------------------------------------------
run_cmd "$VENV_PY" "${PROJECT_ROOT}/scripts/gee/clip_alphaearth_mosaic.py" \
    --input "${PROJECT_ROOT}/data/interim/bd_coastal_alphaearth_2024_mosaic.tif" \
    --output "${PROJECT_ROOT}/data/processed/features/bd_coastal_alphaearth_2024_clipped.tif"

log "Pipeline completed successfully 🎉"
