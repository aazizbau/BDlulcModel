"""Runner utilities for final repeated-seed model training.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Run or summarize repeated-seed experiments for empirical model-performance uncertainty.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--config``, ``--output-root``, ``--seeds``, ``--force``, ``--dry-run``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate selected configurations and seed outputs from the new AOI training data; do not mix summaries from different spatial splits.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Import this helper from its parent workflow or an interactive check::

    import scripts.final_10seed_experiment.common.runner_utils
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from .experiment_constants import (
    DEFAULT_OUTPUT_ROOT,
    EXPERIMENT_NAME,
    MODEL_FAMILY_DISPLAY,
    SEEDS,
    feature_set_for_run_name,
)
from .output_utils import copy_config, write_json

JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config is empty or invalid: {path}")
    return config


def normalize_family(family: str) -> str:
    return str(family).lower().strip()


def run_name_for_seed(model_family: str, feature_set: str, seed: int) -> str:
    family_display = MODEL_FAMILY_DISPLAY.get(normalize_family(model_family), model_family)
    feature_display = feature_set_for_run_name(feature_set)
    return f"final10seed_{family_display}_{feature_display}_seed{seed:03d}"


def seed_output_dir(output_root: Path, model_family: str, seed: int) -> Path:
    family_display = MODEL_FAMILY_DISPLAY.get(normalize_family(model_family), model_family)
    return output_root / family_display / f"seed_{seed:03d}"


def cli_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_arg(command: list[str], key: str, value: Any) -> None:
    if key in {"outdir", "seed"}:
        return
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(cli_name(key))
        return
    command.append(cli_name(key))
    if isinstance(value, list):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def build_command(config: dict[str, Any], seed: int, outdir: Path) -> list[str]:
    args = dict(config["args"])
    args["data"] = config["data"]
    command = [sys.executable, str(resolve_path(Path(config["training_script"])))]
    for key, value in args.items():
        append_arg(command, key, value)
    command.extend(["--outdir", str(outdir), "--seed", str(seed)])
    return command


def write_run_metadata(outdir: Path, config: dict[str, Any], seed: int, run_name: str, command: list[str]) -> None:
    write_json(
        outdir / "run_metadata.json",
        {
            "experiment_name": EXPERIMENT_NAME,
            "created_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
            "run_name": run_name,
            "seed": seed,
            "model_family": config["model_family"],
            "model": config["model"],
            "feature_set": config["feature_set"],
            "source_best_run_name": config["source_best_run_name"],
            "source_best_run_dir": config["source_best_run_dir"],
            "command": command,
        },
    )


def parse_runner_args(model_family: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the final 10-seed best-configuration experiment for {model_family}."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(f"scripts/final_10seed_experiment/configs/{model_family}_best.yaml"),
        help="Frozen selected-configuration YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Fixed seeds to run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when summary.json already exists for a seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training.",
    )
    return parser.parse_args()


def run_family(model_family: str) -> None:
    args = parse_runner_args(model_family)
    config_path = resolve_path(args.config)
    output_root = resolve_path(args.output_root)
    config = load_config(config_path)

    if normalize_family(config["model_family"]) != normalize_family(model_family):
        raise ValueError(
            f"Config model_family={config['model_family']!r} does not match runner {model_family!r}."
        )

    if not args.dry_run:
        copy_config(config_path, output_root, config["model_family"])

    for seed in args.seeds:
        run_name = run_name_for_seed(config["model_family"], config["feature_set"], seed)
        outdir = seed_output_dir(output_root, config["model_family"], seed)
        summary = outdir / "summary.json"
        command = build_command(config, seed, outdir)

        print("=" * 80)
        print(f"RUN: {run_name}")
        print(f"OUT: {outdir}")
        print("CMD:", " ".join(command))

        if summary.exists() and not args.force:
            print(f"SKIP: {summary} already exists.")
            continue

        if args.dry_run:
            continue

        outdir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(outdir, config, seed, run_name, command)
        write_json(outdir / "run_config.json", {**config, "seed": seed, "run_name": run_name})

        log_path = outdir / "run.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(f"Training failed for {run_name}; see {log_path}")
