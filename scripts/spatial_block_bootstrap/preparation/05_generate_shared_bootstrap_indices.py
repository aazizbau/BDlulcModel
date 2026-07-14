#!/usr/bin/env python3
"""Generate one shared set of paired test-block bootstrap indices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.bootstrap_utils import generate_indices  # noqa: E402
from common.constants import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_BOOTSTRAP_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    resolve_path,
)
from common.output_utils import load_yaml, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_BOOTSTRAP_CONFIG)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    config = load_yaml(args.config)
    settings = config["bootstrap"]
    replicates = int(args.bootstrap or settings["replicates"])
    seed = int(settings["random_seed"] if args.seed is None else args.seed)

    validation_path = output_root / "metadata" / "validation_status.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise RuntimeError("Validation has not passed; shared indices were not generated.")

    inventory = pd.read_csv(output_root / "metadata" / "test_block_inventory.csv")
    block_sets = []
    for path in inventory["prediction_path"]:
        actual = Path(path)
        if actual.suffixes[-2:] == [".csv", ".gz"]:
            blocks = set(pd.read_csv(actual, usecols=["block_id"])["block_id"])
        else:
            blocks = set(pd.read_parquet(actual, columns=["block_id"])["block_id"])
        block_sets.append(blocks)
    reference = block_sets[0]
    if any(blocks != reference for blocks in block_sets[1:]):
        raise RuntimeError("Selected runs do not share the same test block ID set.")

    block_ids = sorted(reference)
    index_dir = output_root / "bootstrap_indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"block_position": range(len(block_ids)), "block_id": block_ids}
    ).to_csv(index_dir / "test_block_ids.csv", index=False)
    indices = generate_indices(len(block_ids), replicates, seed)
    np.save(index_dir / "shared_test_block_bootstrap_indices.npy", indices)

    pd.DataFrame(
        [{"class_id": key, "class_name": value} for key, value in CLASS_NAMES.items()]
    ).to_csv(output_root / "metadata" / "class_definitions.csv", index=False)
    final_settings = {
        **config,
        "bootstrap": {**settings, "replicates": replicates, "random_seed": seed},
        "n_test_blocks": len(block_ids),
    }
    write_json(output_root / "metadata" / "bootstrap_settings.json", final_settings)
    print(f"Test blocks: {len(block_ids):,}")
    print(f"Bootstrap replicates: {replicates:,}")
    print(f"Saved: {index_dir / 'shared_test_block_bootstrap_indices.npy'}")


if __name__ == "__main__":
    main()
