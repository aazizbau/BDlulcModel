#!/usr/bin/env python3
"""
Run all six final 10-seed best-configuration experiments.

Complete Example Run
--------------------
python scripts/final_10seed_experiment/run_all_10seed_experiments.py \
    --output-root outputs/final_10seed_experiment \
    --resume

Resume Support
--------------
Resume mode is enabled by default. A model-family/seed run is considered
complete when its ``summary.json`` exists, and completed runs are skipped.
Use ``--force`` to rerun every requested seed.

Each non-dry run writes a timestamped orchestration log to:

    outputs/final_10seed_experiment/logs/
    run_all_10seed_experiments_<YYYYMMDD_HHMMSS>.log

Dry Run
-------
python scripts/final_10seed_experiment/run_all_10seed_experiments.py \
    --output-root outputs/final_10seed_experiment \
    --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common.experiment_constants import (
    DEFAULT_OUTPUT_ROOT,
    MODEL_FAMILY_DISPLAY,
    MODEL_FAMILY_ORDER,
    SEEDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))


class JSTFormatter(logging.Formatter):
    """Format log timestamps in Japan Standard Time."""

    def converter(self, timestamp: float) -> tuple:
        return datetime.fromtimestamp(timestamp, JST).timetuple()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_logger(output_root: Path, dry_run: bool) -> tuple[logging.Logger, Path]:
    run_started = datetime.now(JST)
    log_path = (
        output_root
        / "logs"
        / f"run_all_10seed_experiments_{run_started:%Y%m%d_%H%M%S}.log"
    )

    logger = logging.getLogger("final_10seed_experiment")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = JSTFormatter(
        fmt="%(asctime)s JST | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if not dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, log_path


def completed_seed(output_root: Path, family: str, seed: int) -> bool:
    family_dir = MODEL_FAMILY_DISPLAY[family]
    return (output_root / family_dir / f"seed_{seed:03d}" / "summary.json").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all final 10-seed model-family experiments.")
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume",
        dest="force",
        action="store_false",
        help="Skip seeds whose summary.json already exists (default).",
    )
    run_mode.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Rerun all requested model-family/seed combinations.",
    )
    parser.set_defaults(force=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    logger, log_path = configure_logger(output_root, args.dry_run)

    logger.info("Starting final 10-seed experiment orchestration.")
    logger.info("Output root: %s", output_root)
    logger.info("Requested seeds: %s", ", ".join(str(seed) for seed in args.seeds))
    logger.info("Resume mode: %s", "disabled (--force)" if args.force else "enabled")
    if args.dry_run:
        logger.info("Dry run: no log file or training outputs will be written.")
        logger.info("A real run would save its log under: %s", log_path.parent)
    else:
        logger.info("Orchestration log: %s", log_path)

    for family in MODEL_FAMILY_ORDER:
        seeds_to_run = list(args.seeds)
        if not args.force:
            completed = [
                seed for seed in args.seeds if completed_seed(output_root, family, seed)
            ]
            seeds_to_run = [seed for seed in args.seeds if seed not in completed]
            if completed:
                logger.info(
                    "%s: resuming; completed seeds skipped: %s",
                    MODEL_FAMILY_DISPLAY[family],
                    ", ".join(str(seed) for seed in completed),
                )
            if not seeds_to_run:
                logger.info(
                    "%s: all requested seeds are complete; skipping family.",
                    MODEL_FAMILY_DISPLAY[family],
                )
                continue

        script = (
            PROJECT_ROOT
            / "scripts"
            / "final_10seed_experiment"
            / f"train_{family}_10seeds.py"
        )
        if family == "lgbm":
            script = (
                PROJECT_ROOT
                / "scripts"
                / "final_10seed_experiment"
                / "train_lightgbm_10seeds.py"
            )
        command = [
            sys.executable,
            "-u",
            str(script),
            "--output-root",
            str(output_root),
            "--seeds",
            *[str(seed) for seed in seeds_to_run],
        ]
        if args.force:
            command.append("--force")
        if args.dry_run:
            command.append("--dry-run")
        logger.info("%s", "=" * 80)
        logger.info("Starting model family: %s", MODEL_FAMILY_DISPLAY[family])
        logger.info("Command: %s", " ".join(command))

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            logger.info("[%s] %s", MODEL_FAMILY_DISPLAY[family], line.rstrip())
        returncode = process.wait()
        if returncode != 0:
            logger.error(
                "%s failed with exit code %d. Rerun with --resume after correcting the cause.",
                MODEL_FAMILY_DISPLAY[family],
                returncode,
            )
            raise SystemExit(returncode)
        logger.info("Completed model family: %s", MODEL_FAMILY_DISPLAY[family])

    logger.info("All requested model-family/seed runs are complete.")


if __name__ == "__main__":
    main()
