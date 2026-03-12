#!/usr/bin/env python3
"""
Check AE64 + 10 indices training NPZ contents.

This script validates and summarizes an NPZ file produced by:
- extract_ae64_plus10indices_samples.py
- extract_ae64_plus10indices_samples_trainvaltest.py

It supports both:
- train/val NPZ
- train/val/test NPZ

It reports:
- available keys
- array shapes and dtypes
- feature names
- metadata summary
- class counts
- realized split fractions
- normalization sanity checks for mu and sigma

Examples:
  python scripts/training/check_ae64_plus10indices_npz.py \
    --npz data/processed/training/ae64_plus10indices_samples_4upazila_2023.npz

  python scripts/training/check_ae64_plus10indices_npz.py \
    --npz data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz

  python scripts/training/check_ae64_plus10indices_npz.py \
    --npz data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz \
    --show-all-feature-names
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_NUM_CLASSES = 10


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}", flush=True)


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


def class_counts_from_y(y: np.ndarray) -> Dict[int, int]:
    if y.size == 0:
        return {}
    unique, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def label_presence(y: np.ndarray) -> Dict[str, list[int]]:
    present = sorted(np.unique(y).astype(int).tolist()) if y.size > 0 else []
    missing = [c for c in range(1, EXPECTED_NUM_CLASSES + 1) if c not in present]
    invalid = [c for c in present if c < 1 or c > EXPECTED_NUM_CLASSES]
    return {
        "present": present,
        "missing": missing,
        "invalid": invalid,
    }


def maybe_get_array(d: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    return d[key] if key in d.files else None


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
        X_test = maybe_get_array(d, "X_test")
        y_test = maybe_get_array(d, "y_test")
        mu = d["mu"]
        sigma = d["sigma"]
        feature_names = d["feature_names"]

        try:
            meta = json.loads(str(d["meta"]))
        except Exception as e:
            raise SystemExit(f"Failed to parse meta JSON: {e}")

        has_test = (X_test is not None) and (y_test is not None)

        log(f"X_train: shape={X_train.shape}, dtype={X_train.dtype}")
        log(f"X_val  : shape={X_val.shape}, dtype={X_val.dtype}")
        if has_test:
            log(f"X_test : shape={X_test.shape}, dtype={X_test.dtype}")
        log(f"y_train: shape={y_train.shape}, dtype={y_train.dtype}")
        log(f"y_val  : shape={y_val.shape}, dtype={y_val.dtype}")
        if has_test:
            log(f"y_test : shape={y_test.shape}, dtype={y_test.dtype}")
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
        log(f"fusion_type (meta): {meta.get('fusion_type', None)}")
        log(f"upazilas (meta)   : {meta.get('upazilas', None)}")

        final_counts = meta.get("final_counts", {})
        train_per_class_meta = final_counts.get("train_per_class", {})
        val_per_class_meta = final_counts.get("val_per_class", {})
        test_per_class_meta = final_counts.get("test_per_class", {}) if has_test else {}

        log(f"train_per_class (meta): {train_per_class_meta}")
        log(f"val_per_class   (meta): {val_per_class_meta}")
        if has_test:
            log(f"test_per_class  (meta): {test_per_class_meta}")

        errors: list[str] = []

        # Structural checks
        if X_train.ndim != 2:
            errors.append(f"X_train must be 2D, got ndim={X_train.ndim}")
        if X_val.ndim != 2:
            errors.append(f"X_val must be 2D, got ndim={X_val.ndim}")
        if has_test and X_test is not None and X_test.ndim != 2:
            errors.append(f"X_test must be 2D, got ndim={X_test.ndim}")

        if y_train.ndim != 1:
            errors.append(f"y_train must be 1D, got ndim={y_train.ndim}")
        if y_val.ndim != 1:
            errors.append(f"y_val must be 1D, got ndim={y_val.ndim}")
        if has_test and y_test is not None and y_test.ndim != 1:
            errors.append(f"y_test must be 1D, got ndim={y_test.ndim}")

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
        if has_test and X_test is not None and y_test is not None and X_test.shape[0] != y_test.shape[0]:
            errors.append(
                f"Row mismatch: X_test has {X_test.shape[0]} rows but y_test has {y_test.shape[0]}"
            )

        if X_train.shape[1] != len(feature_names):
            errors.append(
                f"Feature mismatch: X_train width={X_train.shape[1]} but len(feature_names)={len(feature_names)}"
            )
        if X_val.shape[1] != len(feature_names):
            errors.append(
                f"Feature mismatch: X_val width={X_val.shape[1]} but len(feature_names)={len(feature_names)}"
            )
        if has_test and X_test is not None and X_test.shape[1] != len(feature_names):
            errors.append(
                f"Feature mismatch: X_test width={X_test.shape[1]} but len(feature_names)={len(feature_names)}"
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

        # Finite checks
        X_train_finite = bool(np.isfinite(X_train).all())
        X_val_finite = bool(np.isfinite(X_val).all())
        X_test_finite = bool(np.isfinite(X_test).all()) if has_test and X_test is not None else True
        mu_finite = bool(np.isfinite(mu).all())
        sigma_finite = bool(np.isfinite(sigma).all())

        log(f"X_train finite       : {X_train_finite}")
        log(f"X_val finite         : {X_val_finite}")
        if has_test:
            log(f"X_test finite        : {X_test_finite}")
        log(f"mu finite            : {mu_finite}")
        log(f"sigma finite         : {sigma_finite}")

        if not X_train_finite:
            errors.append("X_train contains non-finite values")
        if not X_val_finite:
            errors.append("X_val contains non-finite values")
        if has_test and not X_test_finite:
            errors.append("X_test contains non-finite values")
        if not mu_finite:
            errors.append("mu contains non-finite values")
        if not sigma_finite:
            errors.append("sigma contains non-finite values")

        # Sigma checks
        sigma_nonpositive_count = int((sigma <= 0).sum())
        sigma_zero_count = int((sigma == 0).sum())
        sigma_min = float(np.min(sigma))
        sigma_max = float(np.max(sigma))

        log(f"sigma <= 0 count     : {sigma_nonpositive_count}")
        log(f"sigma == 0 count     : {sigma_zero_count}")
        log(f"sigma min            : {sigma_min}")
        log(f"sigma max            : {sigma_max}")

        if sigma_nonpositive_count > 0:
            errors.append(f"sigma has {sigma_nonpositive_count} non-positive values")

        # Label summary from arrays
        train_counts_direct = class_counts_from_y(y_train)
        val_counts_direct = class_counts_from_y(y_val)
        test_counts_direct = class_counts_from_y(y_test) if has_test and y_test is not None else {}

        log(f"train_per_class (direct from y_train): {train_counts_direct}")
        log(f"val_per_class   (direct from y_val)  : {val_counts_direct}")
        if has_test:
            log(f"test_per_class  (direct from y_test) : {test_counts_direct}")

        train_presence = label_presence(y_train)
        val_presence = label_presence(y_val)
        test_presence = label_presence(y_test) if has_test and y_test is not None else None

        log(
            f"train labels present={train_presence['present']} "
            f"missing={train_presence['missing']} invalid={train_presence['invalid']}"
        )
        log(
            f"val labels present  ={val_presence['present']} "
            f"missing={val_presence['missing']} invalid={val_presence['invalid']}"
        )
        if has_test and test_presence is not None:
            log(
                f"test labels present ={test_presence['present']} "
                f"missing={test_presence['missing']} invalid={test_presence['invalid']}"
            )

        if train_presence["invalid"]:
            errors.append(f"y_train contains invalid labels: {train_presence['invalid']}")
        if val_presence["invalid"]:
            errors.append(f"y_val contains invalid labels: {val_presence['invalid']}")
        if has_test and test_presence is not None and test_presence["invalid"]:
            errors.append(f"y_test contains invalid labels: {test_presence['invalid']}")

        # Totals and realized fractions
        total_train = int(y_train.size)
        total_val = int(y_val.size)
        total_test = int(y_test.size) if has_test and y_test is not None else 0
        total_all = total_train + total_val + total_test

        val_frac_realized = (total_val / total_all) if total_all > 0 else float("nan")
        test_frac_realized = (total_test / total_all) if total_all > 0 else float("nan")
        train_frac_realized = (total_train / total_all) if total_all > 0 else float("nan")

        log(f"train total          : {total_train}")
        log(f"val total            : {total_val}")
        if has_test:
            log(f"test total           : {total_test}")
        log(f"all total            : {total_all}")
        log(f"realized train frac  : {train_frac_realized:.6f}")
        log(f"realized val fraction: {val_frac_realized:.6f}")
        if has_test:
            log(f"realized test frac   : {test_frac_realized:.6f}")

        # Metadata total checks
        train_total_meta = final_counts.get("train_total", None)
        val_total_meta = final_counts.get("val_total", None)
        test_total_meta = final_counts.get("test_total", None)

        log(f"train_total (meta)   : {train_total_meta}")
        log(f"val_total (meta)     : {val_total_meta}")
        if has_test:
            log(f"test_total (meta)    : {test_total_meta}")

        if train_total_meta is not None and int(train_total_meta) != total_train:
            errors.append(f"Meta train_total={train_total_meta} but actual train total={total_train}")
        if val_total_meta is not None and int(val_total_meta) != total_val:
            errors.append(f"Meta val_total={val_total_meta} but actual val total={total_val}")
        if has_test and test_total_meta is not None and int(test_total_meta) != total_test:
            errors.append(f"Meta test_total={test_total_meta} but actual test total={total_test}")

        # Optional direct-vs-meta class count checks
        def normalize_meta_counts(meta_counts: Dict) -> Dict[int, int]:
            out: Dict[int, int] = {}
            for k, v in meta_counts.items():
                out[int(k)] = int(v)
            return out

        if train_per_class_meta:
            train_per_class_meta_norm = normalize_meta_counts(train_per_class_meta)
            if train_per_class_meta_norm != train_counts_direct:
                errors.append("train_per_class in meta does not match direct y_train counts")

        if val_per_class_meta:
            val_per_class_meta_norm = normalize_meta_counts(val_per_class_meta)
            if val_per_class_meta_norm != val_counts_direct:
                errors.append("val_per_class in meta does not match direct y_val counts")

        if has_test and test_per_class_meta:
            test_per_class_meta_norm = normalize_meta_counts(test_per_class_meta)
            if test_per_class_meta_norm != test_counts_direct:
                errors.append("test_per_class in meta does not match direct y_test counts")

        if errors:
            log("CHECK RESULT: FAILED")
            for err in errors:
                log(f"ERROR: {err}")
            raise SystemExit(1)

        log("CHECK RESULT: PASSED")


if __name__ == "__main__":
    main()
