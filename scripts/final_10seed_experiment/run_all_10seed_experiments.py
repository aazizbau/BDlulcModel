#!/usr/bin/env python3
"""
Run all six final 10-seed best-configuration experiments.

Complete Example Run
--------------------
python scripts/final_10seed_experiment/run_all_10seed_experiments.py \
    --output-root outputs/final_10seed_experiment

Dry Run
-------
python scripts/final_10seed_experiment/run_all_10seed_experiments.py \
    --output-root outputs/final_10seed_experiment \
    --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.experiment_constants import DEFAULT_OUTPUT_ROOT, MODEL_FAMILY_ORDER, SEEDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all final 10-seed model-family experiments.")
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for family in MODEL_FAMILY_ORDER:
        script = PROJECT_ROOT / "scripts" / "final_10seed_experiment" / f"train_{family}_10seeds.py"
        if family == "lgbm":
            script = PROJECT_ROOT / "scripts" / "final_10seed_experiment" / "train_lightgbm_10seeds.py"
        command = [
            sys.executable,
            str(script),
            "--output-root",
            str(args.output_root),
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        if args.force:
            command.append("--force")
        if args.dry_run:
            command.append("--dry-run")
        print("=" * 80)
        print("CMD:", " ".join(command))
        process = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if process.returncode != 0:
            raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
