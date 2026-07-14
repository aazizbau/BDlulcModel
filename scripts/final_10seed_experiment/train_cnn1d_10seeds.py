#!/usr/bin/env python3
"""
Run the final 10-seed experiment for the selected best CNN1D configuration.

Complete Example Run
--------------------
python scripts/final_10seed_experiment/train_cnn1d_10seeds.py \
    --config scripts/final_10seed_experiment/configs/cnn1d_best.yaml \
    --output-root outputs/final_10seed_experiment
"""

from common.runner_utils import run_family


if __name__ == "__main__":
    run_family("cnn1d")

