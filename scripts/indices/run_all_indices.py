"""
Run all index scripts for a given year.

Example:
    python scripts/indices/run_all_indices.py --year 2023
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


INDEX_SCRIPTS = [
    "make_ndvi.py",
    "make_nirv.py",
    "make_ndwi.py",
    "make_msavi.py",
    "make_ndmi.py",
    "make_ndpi.py",
    "make_bsi.py",
    "make_ndbi.py",
    "make_awei_sh.py",
    "make_evi.py",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all index scripts for a year.")
    parser.add_argument("--year", type=int, required=True, help="Target year.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent

    for script_name in INDEX_SCRIPTS:
        script_path = script_dir / script_name
        if not script_path.exists():
            raise SystemExit(f"Missing script: {script_path}")

        cmd = [sys.executable, str(script_path), "--year", str(args.year)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
