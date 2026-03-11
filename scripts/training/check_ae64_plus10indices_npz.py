#!/usr/bin/env python3
"""
Check AE64 + 10 indices training NPZ contents.

This script validates and summarizes an NPZ file produced by
extract_ae64_plus10indices_samples.py.

It reports:
- available keys
- array shapes and dtypes
- feature names
- metadata summary
- class counts
- normalization sanity checks for mu and sigma

Example:
  python scripts/training/check_ae64_plus10indices_npz.py \
    --npz data/processed/training/ae64_plus10indices_samples_4upazila_2023.npz

python scripts/training/check_ae64_plus10indices_npz.py \
  --npz data/processed/training/ae64_plus10indices_samples_4upazila_2023.npz \
  --show-all-feature-names
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check AE64 + 10 indices training NPZ contents and normalization stats."
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=Path("data/processed/training/ae64_plus10indices_samples_4upazila_2023.npz"),
        help="Path to NPZ file to inspect.",
    )
    p.add_argument(
        "--show-all-feature-names",
        action="store_true",
        help="Print all feature names instead of only summary/count.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_path(args.npz)

    if not npz_path.exists():
        raise SystemExit(f"NPZ file not found: {npz_path}")

    log(f"Checking NPZ: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as d:
        log(f"keys: {list(d.files)}")

        required_keys = [
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "mu",
            "sigma",
            "feature_names",
            "meta",
        ]
        missing = [k for k in required_keys if k not in d.files]
        if missing:
            raise SystemExit(f"Missing required keys: {missing}")

        X_train = d["X_train"]
        y_train = d["y_train"]
        X_val = d["X_val"]
        y_val = d["y_val"]
        mu = d["mu"]
        sigma = d["sigma"]
        feature_names = d["feature_names"]

        try:
            meta = json.loads(str(d["meta"]))
        except Exception as e:
            raise SystemExit(f"Failed to parse meta JSON: {e}")

        log(f"X_train: shape={X_train.shape}, dtype={X_train.dtype}")
        log(f"X_val  : shape={X_val.shape}, dtype={X_val.dtype}")
        log(f"y_train: shape={y_train.shape}, dtype={y_train.dtype}")
        log(f"y_val  : shape={y_val.shape}, dtype={y_val.dtype}")
        log(f"mu     : shape={mu.shape}, dtype={mu.dtype}")
        log(f"sigma  : shape={sigma.shape}, dtype={sigma.dtype}")

        log(f"feature_names count: {len(feature_names)}")
        if args.show_all_feature_names:
            log(f"feature_names: {feature_names.tolist()}")
        else:
            preview_head = feature_names[:10].tolist()
            preview_tail = feature_names[-10:].tolist()
            log(f"feature_names head(10): {preview_head}")
            log(f"feature_names tail(10): {preview_tail}")

        feature_dim_meta = meta.get("feature_dim", None)
        log(f"feature_dim (meta): {feature_dim_meta}")

        train_per_class = meta.get("final_counts", {}).get("train_per_class", {})
        val_per_class = meta.get("final_counts", {}).get("val_per_class", {})
        log(f"train_per_class: {train_per_class}")
        log(f"val_per_class  : {val_per_class}")

        # Structural checks
        errors: list[str] = []

        if X_train.ndim != 2:
            errors.append(f"X_train must be 2D, got ndim={X_train.ndim}")
        if X_val.ndim != 2:
            errors.append(f"X_val must be 2D, got ndim={X_val.ndim}")
        if y_train.ndim != 1:
            errors.append(f"y_train must be 1D, got ndim={y_train.ndim}")
        if y_val.ndim != 1:
            errors.append(f"y_val must be 1D, got ndim={y_val.ndim}")
        if mu.ndim != 1:
            errors.append(f"mu must be 1D, got ndim={mu.ndim}")
        if sigma.ndim != 1:
            errors.append(f"sigma must be 1D, got ndim={sigma.ndim}")

        if X_train.shape[0] != y_train.shape[0]:
            errors.append(
                f"Row mismatch: X_train has {X_train.shape[0]} rows but y_train has {y_train.shape[0]}"
            )
        if X_val.shape[0] != y_val.shape[0]:
            errors.append(
                f"Row mismatch: X_val has {X_val.shape[0]} rows but y_val has {y_val.shape[0]}"
            )

        if X_train.shape[1] != len(feature_names):
            errors.append(
                f"Feature mismatch: X_train width={X_train.shape[1]} but len(feature_names)={len(feature_names)}"
            )
        if X_val.shape[1] != len(feature_names):
            errors.append(
                f"Feature mismatch: X_val width={X_val.shape[1]} but len(feature_names)={len(feature_names)}"
            )
        if mu.shape[0] != len(feature_names):
            errors.append(
                f"mu length mismatch: mu.shape[0]={mu.shape[0]} but len(feature_names)={len(feature_names)}"
            )
        if sigma.shape[0] != len(feature_names):
            errors.append(
                f"sigma length mismatch: sigma.shape[0]={sigma.shape[0]} but len(feature_names)={len(feature_names)}"
            )

        if feature_dim_meta is not None and int(feature_dim_meta) != len(feature_names):
            errors.append(
                f"Meta feature_dim={feature_dim_meta} but len(feature_names)={len(feature_names)}"
            )

        # Normalization checks
        mu_finite = bool(np.isfinite(mu).all())
        sigma_finite = bool(np.isfinite(sigma).all())
        sigma_nonpositive_count = int((sigma <= 0).sum())
        sigma_min = float(sigma.min())
        sigma_max = float(sigma.max())

        log(f"mu finite            : {mu_finite}")
        log(f"sigma finite         : {sigma_finite}")
        log(f"sigma <= 0 count     : {sigma_nonpositive_count}")
        log(f"sigma min            : {sigma_min}")
        log(f"sigma max            : {sigma_max}")

        if not mu_finite:
            errors.append("mu contains non-finite values")
        if not sigma_finite:
            errors.append("sigma contains non-finite values")
        if sigma_nonpositive_count > 0:
            errors.append(f"sigma has {sigma_nonpositive_count} non-positive values")

        # Basic label summary
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_val, counts_val = np.unique(y_val, return_counts=True)

        train_counts_direct = {int(k): int(v) for k, v in zip(unique_train, counts_train)}
        val_counts_direct = {int(k): int(v) for k, v in zip(unique_val, counts_val)}

        log(f"train_per_class (direct from y_train): {train_counts_direct}")
        log(f"val_per_class   (direct from y_val)  : {val_counts_direct}")

        total_train = int(y_train.size)
        total_val = int(y_val.size)
        total_all = total_train + total_val
        val_frac_realized = (total_val / total_all) if total_all > 0 else float("nan")

        log(f"train total          : {total_train}")
        log(f"val total            : {total_val}")
        log(f"all total            : {total_all}")
        log(f"realized val fraction: {val_frac_realized:.6f}")

        if errors:
            log("CHECK RESULT: FAILED")
            for err in errors:
                log(f"ERROR: {err}")
            raise SystemExit(1)

        log("CHECK RESULT: PASSED")


if __name__ == "__main__":
    main()
