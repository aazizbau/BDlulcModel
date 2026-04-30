#!/usr/bin/env bash
set -euo pipefail

for seed in $(seq 1 300); do
    output="outputs/figures/bd_coastal_infer_lulc_2017_zoom_seed_${seed}.png"
    echo "Seed ${seed}/300 → ${output}"
    python scripts/poster/make_infer_lulc_map_with_zoom.py \
        --year 2017 \
        --seed "${seed}" \
        --zoom-window-km 5 \
        --zoom-inset-x-frac 0.49 \
        --zoom-inset-y-frac 0.075 \
        --output "${output}"
done

echo "Done. 300 images saved to outputs/figures/"
