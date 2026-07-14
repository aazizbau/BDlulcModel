"""Stable file and identifier naming helpers."""

from __future__ import annotations

import re


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def block_id(upazila: str, block_row: int, block_col: int) -> str:
    return f"{upazila}:r{block_row:04d}_c{block_col:04d}"
