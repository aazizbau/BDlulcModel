"""Output helpers for final repeated-seed experiment scripts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def copy_config(config_path: Path, output_root: Path, family: str) -> None:
    target = output_root / family / "config" / "selected_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, target)

