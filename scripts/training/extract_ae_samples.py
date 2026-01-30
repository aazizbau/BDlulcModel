#!/usr/bin/env python3
"""
Extract balanced pixel samples for training from AlphaEarth 64D mosaic + label rasters.

Example:
  python scripts/training/extract_ae_samples.py \
    --ae data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif \
    --output data/processed/training/ae64_samples_4upazila_2023.npz \
    --max-per-class 300000 \
    --val-frac 0.2 \
    --block-size-m 1000

- Inputs:
  * AlphaEarth mosaic GeoTIFF (64 bands) in EPSG:32646 @ 10m
  * Label rasters (one per upazila) in EPSG:32646 @ 10m, nodata=0, classes 1..10

- Output:
  * NPZ containing X_train, y_train, X_val, y_val (and metadata)

Notes:
- No multiprocessing (keeps CPU usage modest and predictable).
- Uses a spatial "block split" to reduce leakage: blocks are assigned to train/val.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


UPAZILA_LABELS: Dict[str, Path] = {
    "manpura": Path("assets/training_labels/manpura_label_10class_10m.tif"),
    "betagi": Path("assets/training_labels/betagi_label_10class_10m.tif"),
    "amtali": Path("assets/training_labels/amtali_label_10class_10m.tif"),
    "bamna": Path("assets/training_labels/bamna_label_10class_10m.tif"),
}


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract AE64 pixel samples with spatial split + class balancing.")
    p.add_argument(
        "--ae",
        type=Path,
        required=True,
        help="AlphaEarth 64D mosaic GeoTIFF path (EPSG:32646, 10m).",
    )
    p.add_argument(
        "--upazilas",
        nargs="+",
        default=["manpura", "betagi", "amtali", "bamna"],
        help="Which upazilas to use (default: all 4).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/training/ae64_samples_4upazila_2023.npz"),
        help="Output NPZ path.",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=300_000,
        help="Maximum samples per class across ALL upazilas (default: 300k).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Validation fraction using block-based split (default: 0.2).",
    )
    p.add_argument(
        "--block-size-m",
        type=int,
        default=1000,
        help="Spatial block size in meters for split (default: 1000m).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--label-nodata",
        type=int,
        default=0,
        help="Label nodata value (default: 0).",
    )
    p.add_argument(
        "--ae-nodata",
        type=float,
        default=0.0,
        help="AlphaEarth nodata value (default: 0).",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=1024,
        help="Processing chunk size in pixels (default: 1024).",
    )
    return p.parse_args()


def world_to_pixel(transform: rasterio.Affine, x: float, y: float) -> Tuple[int, int]:
    """Return (col,row) pixel indices for world coords."""
    col, row = ~transform * (x, y)
    return int(np.floor(col)), int(np.floor(row))


def block_assign(row: int, col: int, block_px: int, seed: int) -> float:
    """
    Deterministic pseudo-random number in [0,1) from a (block) coordinate.
    Used to assign blocks to train/val without storing huge arrays.
    """
    br = row // block_px
    bc = col // block_px
    v = (br * 73856093) ^ (bc * 19349663) ^ (seed * 83492791)
    v = v & 0xFFFFFFFFFFFFFFFF
    return (v % 10_000_000) / 10_000_000.0


def extract_from_upazila(
    ae_src: rasterio.DatasetReader,
    label_path: Path,
    upazila: str,
    label_nodata: int,
    ae_nodata: float,
    val_frac: float,
    block_px: int,
    max_per_class_remaining: Dict[int, int],
    rng: np.random.Generator,
    chunk: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, int]]:
    """
    Returns lists of X_train parts, y_train parts, X_val parts, y_val parts.
    Also returns a small stats dict of accepted samples.
    """
    stats = {"train": 0, "val": 0}
    per_class_taken_train = {k: 0 for k in range(1, 11)}
    per_class_taken_val = {k: 0 for k in range(1, 11)}

    with rasterio.open(label_path) as lbl:
        if lbl.crs != ae_src.crs:
            raise SystemExit(f"[{upazila}] CRS mismatch: label {lbl.crs} vs AE {ae_src.crs}")
        if abs(lbl.transform.a) != abs(ae_src.transform.a) or abs(lbl.transform.e) != abs(ae_src.transform.e):
            raise SystemExit(f"[{upazila}] Resolution mismatch between label and AE.")

        lbl_bounds = lbl.bounds
        ae_bounds = ae_src.bounds

        left = max(lbl_bounds.left, ae_bounds.left)
        right = min(lbl_bounds.right, ae_bounds.right)
        bottom = max(lbl_bounds.bottom, ae_bounds.bottom)
        top = min(lbl_bounds.top, ae_bounds.top)

        if left >= right or bottom >= top:
            raise SystemExit(f"[{upazila}] No spatial overlap between label and AE mosaic.")

        lbl_c0, lbl_r0 = world_to_pixel(lbl.transform, left, top)
        lbl_c1, lbl_r1 = world_to_pixel(lbl.transform, right, bottom)

        ae_c0, ae_r0 = world_to_pixel(ae_src.transform, left, top)
        ae_c1, ae_r1 = world_to_pixel(ae_src.transform, right, bottom)

        lbl_c0, lbl_c1 = sorted((max(0, lbl_c0), min(lbl.width, lbl_c1)))
        lbl_r0, lbl_r1 = sorted((max(0, lbl_r0), min(lbl.height, lbl_r1)))
        ae_c0, ae_c1 = sorted((max(0, ae_c0), min(ae_src.width, ae_c1)))
        ae_r0, ae_r1 = sorted((max(0, ae_r0), min(ae_src.height, ae_r1)))

        h = min(lbl_r1 - lbl_r0, ae_r1 - ae_r0)
        w = min(lbl_c1 - lbl_c0, ae_c1 - ae_c0)

        lbl_win_full = Window(lbl_c0, lbl_r0, w, h)
        ae_win_full = Window(ae_c0, ae_r0, w, h)

        Xtr_parts: List[np.ndarray] = []
        ytr_parts: List[np.ndarray] = []
        Xva_parts: List[np.ndarray] = []
        yva_parts: List[np.ndarray] = []

        for row_off in range(0, int(h), chunk):
            rh = min(chunk, int(h) - row_off)
            for col_off in range(0, int(w), chunk):
                cw = min(chunk, int(w) - col_off)

                lbl_win = Window(lbl_win_full.col_off + col_off, lbl_win_full.row_off + row_off, cw, rh)
                ae_win = Window(ae_win_full.col_off + col_off, ae_win_full.row_off + row_off, cw, rh)

                y = lbl.read(1, window=lbl_win)
                valid_mask = (y != label_nodata) & (y >= 1) & (y <= 10)
                if not np.any(valid_mask):
                    continue

                X = ae_src.read(list(range(1, ae_src.count + 1)), window=ae_win).astype(np.float32)

                if ae_nodata == 0.0:
                    ae_valid = np.any(X != 0.0, axis=0)
                else:
                    ae_valid = np.any(X != float(ae_nodata), axis=0)

                valid_mask &= ae_valid
                if not np.any(valid_mask):
                    continue

                rows, cols = np.where(valid_mask)
                idx = np.arange(rows.size)
                rng.shuffle(idx)
                rows = rows[idx]
                cols = cols[idx]

                xtr_list = []
                ytr_list = []
                xva_list = []
                yva_list = []

                for rr, cc in zip(rows, cols):
                    cls = int(y[rr, cc])
                    if max_per_class_remaining.get(cls, 0) <= 0:
                        continue

                    global_row = int(lbl_win.row_off + rr)
                    global_col = int(lbl_win.col_off + cc)

                    u = block_assign(global_row, global_col, block_px, seed=int(rng.bit_generator._seed_seq.entropy))
                    is_val = u < val_frac

                    feat = X[:, rr, cc]
                    if not np.isfinite(feat).all():
                        continue

                    if is_val:
                        xva_list.append(feat)
                        yva_list.append(cls)
                        stats["val"] += 1
                        per_class_taken_val[cls] += 1
                    else:
                        xtr_list.append(feat)
                        ytr_list.append(cls)
                        stats["train"] += 1
                        per_class_taken_train[cls] += 1

                    max_per_class_remaining[cls] -= 1
                    if max_per_class_remaining[cls] <= 0:
                        max_per_class_remaining[cls] = 0

                if xtr_list:
                    Xtr_parts.append(np.stack(xtr_list).astype(np.float32))
                    ytr_parts.append(np.array(ytr_list, dtype=np.uint8))
                if xva_list:
                    Xva_parts.append(np.stack(xva_list).astype(np.float32))
                    yva_parts.append(np.array(yva_list, dtype=np.uint8))

                if all(v <= 0 for v in max_per_class_remaining.values()):
                    break
            if all(v <= 0 for v in max_per_class_remaining.values()):
                break

    stats.update({f"train_c{c}": per_class_taken_train[c] for c in range(1, 11)})
    stats.update({f"val_c{c}": per_class_taken_val[c] for c in range(1, 11)})
    return Xtr_parts, ytr_parts, Xva_parts, yva_parts, stats


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for u in args.upazilas:
        if u not in UPAZILA_LABELS:
            raise SystemExit(f"Unknown upazila '{u}'. Choose from: {list(UPAZILA_LABELS.keys())}")
        if not UPAZILA_LABELS[u].exists():
            raise SystemExit(f"Missing label raster for {u}: {UPAZILA_LABELS[u]}")

    if not args.ae.exists():
        raise SystemExit(f"AlphaEarth mosaic not found: {args.ae}")

    rng = np.random.default_rng(args.seed)
    max_per_class_remaining = {c: int(args.max_per_class) for c in range(1, 11)}
    block_px = max(1, int(round(args.block_size_m / 10.0)))

    all_Xtr: List[np.ndarray] = []
    all_ytr: List[np.ndarray] = []
    all_Xva: List[np.ndarray] = []
    all_yva: List[np.ndarray] = []
    per_upazila_stats: Dict[str, Dict[str, int]] = {}

    with rasterio.open(args.ae) as ae_src:
        if ae_src.count != 64:
            raise SystemExit(f"Expected 64 bands, found {ae_src.count} in {args.ae}")
        if ae_src.crs is None:
            raise SystemExit("AlphaEarth raster has no CRS.")

        for upazila in args.upazilas:
            label_path = UPAZILA_LABELS[upazila]
            log(f"Extracting from {upazila}: {label_path}")

            Xtr_parts, ytr_parts, Xva_parts, yva_parts, stats = extract_from_upazila(
                ae_src=ae_src,
                label_path=label_path,
                upazila=upazila,
                label_nodata=args.label_nodata,
                ae_nodata=args.ae_nodata,
                val_frac=args.val_frac,
                block_px=block_px,
                max_per_class_remaining=max_per_class_remaining,
                rng=rng,
                chunk=args.chunk,
            )

            if Xtr_parts:
                all_Xtr.append(np.concatenate(Xtr_parts, axis=0))
                all_ytr.append(np.concatenate(ytr_parts, axis=0))
            if Xva_parts:
                all_Xva.append(np.concatenate(Xva_parts, axis=0))
                all_yva.append(np.concatenate(yva_parts, axis=0))

            per_upazila_stats[upazila] = stats

            log(f"{upazila} samples -> train={stats['train']} val={stats['val']}")
            remaining = {k: v for k, v in max_per_class_remaining.items()}
            log(f"Remaining per-class budget: {remaining}")

            if all(v <= 0 for v in max_per_class_remaining.values()):
                log("Reached max-per-class budgets for all classes. Stopping early.")
                break

    if not all_Xtr or not all_ytr:
        raise SystemExit("No training samples extracted. Check overlap / nodata / labels.")

    X_train = np.concatenate(all_Xtr, axis=0).astype(np.float32)
    y_train = np.concatenate(all_ytr, axis=0).astype(np.uint8)

    X_val = np.concatenate(all_Xva, axis=0).astype(np.float32) if all_Xva else np.zeros((0, 64), np.float32)
    y_val = np.concatenate(all_yva, axis=0).astype(np.uint8) if all_yva else np.zeros((0,), np.uint8)

    meta = {
        "ae_path": str(args.ae),
        "upazilas": args.upazilas,
        "label_paths": {u: str(UPAZILA_LABELS[u]) for u in args.upazilas},
        "max_per_class": args.max_per_class,
        "val_frac": args.val_frac,
        "block_size_m": args.block_size_m,
        "block_px": block_px,
        "seed": args.seed,
        "per_upazila_stats": per_upazila_stats,
        "final_counts": {
            "train_total": int(y_train.size),
            "val_total": int(y_val.size),
            "train_per_class": {int(c): int(np.sum(y_train == c)) for c in range(1, 11)},
            "val_per_class": {int(c): int(np.sum(y_val == c)) for c in range(1, 11)},
        },
    }

    np.savez_compressed(
        args.output,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        meta=json.dumps(meta),
    )

    log(f"Saved: {args.output}")
    log(f"Final class counts (train): {meta['final_counts']['train_per_class']}")
    log(f"Final class counts (val)  : {meta['final_counts']['val_per_class']}")


if __name__ == "__main__":
    main()
