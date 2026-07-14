"""Load aligned inputs shared by bootstrap analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_OUTPUT_ROOT, resolve_path
from .output_utils import read_table


def load_bootstrap_context(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, list[str], np.ndarray, dict]:
    root = resolve_path(output_root)
    validation = json.loads(
        (root / "metadata" / "validation_status.json").read_text(encoding="utf-8")
    )
    if not validation.get("passed"):
        raise RuntimeError("Spatial block validation has not passed.")
    selected = pd.read_csv(root / "metadata" / "selected_runs.csv")
    block_long = read_table(
        root
        / "block_confusion_matrices"
        / "all_selected_runs_block_confusion_long.parquet"
    )
    block_ids = pd.read_csv(root / "bootstrap_indices" / "test_block_ids.csv")[
        "block_id"
    ].tolist()
    indices = np.load(root / "bootstrap_indices" / "shared_test_block_bootstrap_indices.npy")
    settings = json.loads(
        (root / "metadata" / "bootstrap_settings.json").read_text(encoding="utf-8")
    )
    return root, selected, block_long, block_ids, indices, settings
