"""I/O helpers with transparent Parquet-to-compressed-CSV fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .constants import resolve_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_table(df: pd.DataFrame, parquet_path: Path) -> Path:
    """Write Parquet when available, otherwise write a compressed CSV."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        fallback = parquet_path.with_suffix(".csv.gz")
        df.to_csv(fallback, index=False, compression="gzip")
        return fallback


def read_table(parquet_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    fallback = parquet_path.with_suffix(".csv.gz")
    if fallback.exists():
        return pd.read_csv(fallback)
    csv_path = parquet_path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Neither {parquet_path}, {fallback}, nor {csv_path} exists."
    )
