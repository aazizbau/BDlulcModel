#!/usr/bin/env python3
"""
Extract balanced pixel samples for training from AlphaEarth 64D mosaic + label rasters (v3, FIXED).

Fixes vs your pasted version:
1) NO hardcoded UPAZILA_LABELS dict anymore.
   - Uses: --labels-dir (default: assets/training_labels_v3)
   - Looks for: <upazila>_label_10class_10m.tif inside that folder
2) Logs the *actual* label paths used (so you can confirm it’s reading _v3 labels).
3) More robust stopping + sanity checks:
   - Stops early if all global targets are satisfied.
   - Warns if an upazila has zero overlap/valid pixels.
4) Keeps your v3 behavior:
   - per-upazila per-class budgets
   - train-only erosion (boundary exclusion) via --erode-px
   - deterministic block hashing split
   - per-class validation constraints through target calculation
   - min_nonzero_bands AE validity

Example:
  python scripts/training/extract_ae_samples_v3.py \
    --ae data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif \
    --labels-dir assets/training_labels_v3 \
    --output data/processed/training/ae64_samples_4upazila_2023_v3.npz \
    --max-per-class-per-upazila 150000 \
    --val-frac 0.2 \
    --min-val-per-class 10000 \
    --min-val-frac-per-class 0.02 \
    --block-size-m 1000 \
    --erode-px 1

Reproduction and AOI adaptation
-------------------------------
Workflow role: Extract spatially split samples, train a classifier, or orchestrate hyperparameter experiments.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--ae``, ``--labels-dir``, ``--upazilas``, ``--output``, ``--max-per-class-per-upazila``, ``--val-frac``, ``--min-val-per-class``, ``--min-val-frac-per-class``, ``--block-size-m``, ``--seed``, ``--label-nodata``, ``--ae-nodata``, ``--chunk``, ``--erode-px``, ``--min-nonzero-bands``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace NPZ/raster/vector inputs with samples extracted from the new AOI, preserve spatially disjoint splits, and review class IDs, feature order, block size, budgets, and random seeds.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
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


# ----------------------------
# Logging / args
# ----------------------------
def log(message: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {message}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract AE64 samples (v3): per-upazila budgets + train erosion + block split + per-class val targets."
    )
    p.add_argument("--ae", type=Path, required=True, help="AlphaEarth 64-band mosaic (EPSG:32646, 10m).")
    p.add_argument("--labels-dir", type=Path, default=Path("assets/training_labels_v3"),
                   help="Directory containing <upazila>_label_10class_10m.tif")
    p.add_argument("--upazilas", nargs="+", default=["manpura", "betagi", "amtali", "bamna"],
                   help="Upazila names; label filenames must match exactly.")
    p.add_argument("--output", type=Path, required=True, help="Output NPZ path.")

    p.add_argument("--max-per-class-per-upazila", type=int, default=150_000,
                   help="Max samples per class per upazila (train+val combined).")
    p.add_argument("--val-frac", type=float, default=0.2, help="Base val fraction (block hashing).")
    p.add_argument("--min-val-per-class", type=int, default=10_000,
                   help="Minimum validation samples per class globally.")
    p.add_argument("--min-val-frac-per-class", type=float, default=0.02,
                   help="Minimum validation fraction per class globally (of total budget).")

    p.add_argument("--block-size-m", type=int, default=1000, help="Block size (meters) for spatial split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument("--label-nodata", type=int, default=0, help="Label nodata value.")
    p.add_argument("--ae-nodata", type=float, default=0.0, help="AE nodata value (0.0 typical).")

    p.add_argument("--chunk", type=int, default=1024, help="Chunk size in pixels for scanning.")
    p.add_argument("--erode-px", type=int, default=1, choices=[0, 1, 2],
                   help="Erode labels for TRAIN only (exclude boundary pixels).")
    p.add_argument("--min-nonzero-bands", type=int, default=8,
                   help="Require >=K nonzero AE bands for valid pixel.")
    return p.parse_args()


# ----------------------------
# Helpers
# ----------------------------
def world_to_pixel(transform: rasterio.Affine, x: float, y: float) -> Tuple[int, int]:
    col, row = ~transform * (x, y)
    return int(np.floor(col)), int(np.floor(row))


def block_assign(row: int, col: int, block_px: int, seed: int) -> float:
    """Deterministic pseudo-random in [0,1) based on block coord."""
    br = row // block_px
    bc = col // block_px
    v = (br * 73856093) ^ (bc * 19349663) ^ (seed * 83492791)
    v = v & 0xFFFFFFFFFFFFFFFF
    return (v % 10_000_000) / 10_000_000.0


def eroded_keep_mask(y: np.ndarray, cls: int, r: int) -> np.ndarray:
    """
    Keep only interior pixels of class 'cls' at erosion radius r.
    For a pixel to be kept, all pixels in its (2r+1)x(2r+1) neighborhood must also be cls.
    """
    if r <= 0:
        return (y == cls)

    pad = r
    yy = np.pad(y, pad_width=pad, mode="constant", constant_values=0)
    keep = np.ones_like(y, dtype=bool)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            view = yy[pad + dy: pad + dy + y.shape[0], pad + dx: pad + dx + y.shape[1]]
            keep &= (view == cls)
            if not keep.any():
                return keep
    return keep


def compute_targets(
    upazilas: List[str],
    max_per_class_per_upazila: int,
    val_frac: float,
    min_val_per_class: int,
    min_val_frac_per_class: float,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Global targets per class based on budgets.
      total_per_class = max_per_class_per_upazila * n_upazilas
      val_target >= max(val_frac*total, min_val_per_class, min_val_frac_per_class*total)
      train_target = total - val_target
    """
    n_u = len(upazilas)
    total_per_class = int(max_per_class_per_upazila) * int(n_u)

    train_targets: Dict[int, int] = {}
    val_targets: Dict[int, int] = {}

    for c in range(1, 11):
        base_val = int(np.ceil(val_frac * total_per_class))
        frac_floor = int(np.ceil(min_val_frac_per_class * total_per_class))
        val_t = max(base_val, int(min_val_per_class), frac_floor)
        val_t = min(val_t, total_per_class)

        train_t = total_per_class - val_t
        val_targets[c] = int(val_t)
        train_targets[c] = int(train_t)

    return train_targets, val_targets


# ----------------------------
# Extraction core
# ----------------------------
def extract_from_upazila(
    ae_src: rasterio.DatasetReader,
    lbl: rasterio.DatasetReader,
    upazila: str,
    label_nodata: int,
    ae_nodata: float,
    min_nonzero_bands: int,
    chunk: int,
    erode_r: int,
    block_px: int,
    seed: int,
    val_frac_base: float,
    upazila_remaining: Dict[int, int],
    train_rem: Dict[int, int],
    val_rem: Dict[int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, int]]:

    stats = {"train": 0, "val": 0}
    per_class_train = {c: 0 for c in range(1, 11)}
    per_class_val = {c: 0 for c in range(1, 11)}

    # Compute overlap bounds in world coords
    lbl_bounds = lbl.bounds
    ae_bounds = ae_src.bounds
    left = max(lbl_bounds.left, ae_bounds.left)
    right = min(lbl_bounds.right, ae_bounds.right)
    bottom = max(lbl_bounds.bottom, ae_bounds.bottom)
    top = min(lbl_bounds.top, ae_bounds.top)

    if left >= right or bottom >= top:
        log(f"[WARN] [{upazila}] No overlap between label and AE mosaic.")
        stats.update({f"train_c{c}": 0 for c in range(1, 11)})
        stats.update({f"val_c{c}": 0 for c in range(1, 11)})
        return [], [], [], [], stats

    # Convert overlap to pixel windows in label and AE
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
    if h <= 0 or w <= 0:
        log(f"[WARN] [{upazila}] Invalid overlap dims (h={h}, w={w}).")
        stats.update({f"train_c{c}": 0 for c in range(1, 11)})
        stats.update({f"val_c{c}": 0 for c in range(1, 11)})
        return [], [], [], [], stats

    lbl_win_full = Window(lbl_c0, lbl_r0, w, h)
    ae_win_full = Window(ae_c0, ae_r0, w, h)

    Xtr_parts: List[np.ndarray] = []
    ytr_parts: List[np.ndarray] = []
    Xva_parts: List[np.ndarray] = []
    yva_parts: List[np.ndarray] = []

    pad = int(erode_r)

    for row_off in range(0, int(h), int(chunk)):
        rh = min(int(chunk), int(h) - row_off)
        for col_off in range(0, int(w), int(chunk)):
            cw = min(int(chunk), int(w) - col_off)

            # If upazila budget is exhausted, stop quickly
            if all(v <= 0 for v in upazila_remaining.values()):
                break

            # If global targets are satisfied, stop everything
            if all(train_rem[c] <= 0 and val_rem[c] <= 0 for c in range(1, 11)):
                break

            # Chunk origin in label pixels
            r0 = int(lbl_win_full.row_off + row_off)
            c0 = int(lbl_win_full.col_off + col_off)

            # Padded label window (to support erosion neighborhood)
            r0p = max(0, r0 - pad)
            c0p = max(0, c0 - pad)
            r1p = min(lbl.height, r0 + rh + pad)
            c1p = min(lbl.width, c0 + cw + pad)
            hp = r1p - r0p
            wp = c1p - c0p
            lbl_win_p = Window(c0p, r0p, wp, hp)

            # Matching padded AE window
            ae_r0_ = int(ae_win_full.row_off + row_off)
            ae_c0_ = int(ae_win_full.col_off + col_off)
            ae_r0p = max(0, ae_r0_ - pad)
            ae_c0p = max(0, ae_c0_ - pad)
            ae_r1p = min(ae_src.height, ae_r0_ + rh + pad)
            ae_c1p = min(ae_src.width, ae_c0_ + cw + pad)
            ae_win_p = Window(ae_c0p, ae_r0p, ae_c1p - ae_c0p, ae_r1p - ae_r0p)

            # Read padded data
            y_pad = lbl.read(1, window=lbl_win_p)
            if y_pad.size == 0:
                continue

            X_pad = ae_src.read(list(range(1, ae_src.count + 1)), window=ae_win_p).astype(np.float32)
            if X_pad.shape[1] != y_pad.shape[0] or X_pad.shape[2] != y_pad.shape[1]:
                # window mismatch (shouldn't happen, but keep safe)
                continue

            # AE validity: require >=K non-nodata/nonzero bands
            if float(ae_nodata) == 0.0:
                nz = (X_pad != 0.0)
            else:
                nz = (X_pad != float(ae_nodata))
            ae_valid = (np.sum(nz, axis=0) >= int(min_nonzero_bands))

            # Slice central (non-padded) chunk region
            rr0 = r0 - r0p
            cc0 = c0 - c0p
            y = y_pad[rr0: rr0 + rh, cc0: cc0 + cw]
            ae_valid_c = ae_valid[rr0: rr0 + rh, cc0: cc0 + cw]

            # Label validity
            valid_lbl = (y != int(label_nodata)) & (y >= 1) & (y <= 10)
            if not np.any(valid_lbl):
                continue
            valid_lbl &= ae_valid_c
            if not np.any(valid_lbl):
                continue

            # Train erosion keep mask (computed from padded labels, sliced to central region)
            train_keep = np.ones_like(y, dtype=bool)
            if pad > 0:
                train_keep[:] = False
                for cls in range(1, 11):
                    keep_pad = eroded_keep_mask(y_pad, cls, pad)
                    keep_c = keep_pad[rr0: rr0 + rh, cc0: cc0 + cw]
                    train_keep |= keep_c

            # Candidate pixels
            rows, cols = np.where(valid_lbl)
            if rows.size == 0:
                continue

            # Deterministic shuffle per chunk
            # (avoids global RNG state and keeps runs repeatable)
            rng = np.random.default_rng((seed * 1000003) ^ (r0 * 9176) ^ (c0 * 6361))
            idx = np.arange(rows.size)
            rng.shuffle(idx)
            rows = rows[idx]
            cols = cols[idx]

            xtr_list: List[np.ndarray] = []
            ytr_list: List[int] = []
            xva_list: List[np.ndarray] = []
            yva_list: List[int] = []

            for rr, cc in zip(rows, cols):
                cls = int(y[rr, cc])

                if upazila_remaining.get(cls, 0) <= 0:
                    continue

                if train_rem.get(cls, 0) <= 0 and val_rem.get(cls, 0) <= 0:
                    continue

                global_row = int(r0 + rr)
                global_col = int(c0 + cc)
                u = block_assign(global_row, global_col, int(block_px), seed=int(seed))
                prefer_val = (u < float(val_frac_base))

                # Decide val/train while respecting remaining targets
                if val_rem[cls] > 0 and train_rem[cls] <= 0:
                    is_val = True
                elif train_rem[cls] > 0 and val_rem[cls] <= 0:
                    is_val = False
                else:
                    is_val = bool(prefer_val)

                # Apply erosion only for TRAIN
                if (not is_val) and pad > 0 and (not train_keep[rr, cc]):
                    continue

                feat = X_pad[:, rr0 + rr, cc0 + cc]
                if not np.isfinite(feat).all():
                    continue

                if is_val:
                    if val_rem[cls] <= 0:
                        continue
                    xva_list.append(feat)
                    yva_list.append(cls)
                    val_rem[cls] -= 1
                    per_class_val[cls] += 1
                    stats["val"] += 1
                else:
                    if train_rem[cls] <= 0:
                        continue
                    xtr_list.append(feat)
                    ytr_list.append(cls)
                    train_rem[cls] -= 1
                    per_class_train[cls] += 1
                    stats["train"] += 1

                upazila_remaining[cls] -= 1

                if all(v <= 0 for v in upazila_remaining.values()):
                    break
                if all(train_rem[c] <= 0 and val_rem[c] <= 0 for c in range(1, 11)):
                    break

            if xtr_list:
                Xtr_parts.append(np.stack(xtr_list).astype(np.float32))
                ytr_parts.append(np.array(ytr_list, dtype=np.uint8))
            if xva_list:
                Xva_parts.append(np.stack(xva_list).astype(np.float32))
                yva_parts.append(np.array(yva_list, dtype=np.uint8))

        if all(v <= 0 for v in upazila_remaining.values()):
            break
        if all(train_rem[c] <= 0 and val_rem[c] <= 0 for c in range(1, 11)):
            break

    stats.update({f"train_c{c}": int(per_class_train[c]) for c in range(1, 11)})
    stats.update({f"val_c{c}": int(per_class_val[c]) for c in range(1, 11)})
    return Xtr_parts, ytr_parts, Xva_parts, yva_parts, stats


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.ae.exists():
        raise SystemExit(f"AlphaEarth mosaic not found: {args.ae}")

    if not args.labels_dir.exists():
        raise SystemExit(f"labels-dir not found: {args.labels_dir}")

    # Build label paths (no dict; derived from labels-dir)
    label_paths: Dict[str, Path] = {}
    for u in args.upazilas:
        lp = args.labels_dir / f"{u}_label_10class_10m.tif"
        if not lp.exists():
            raise SystemExit(f"Missing label raster for {u}: {lp}")
        label_paths[u] = lp

    train_targets, val_targets = compute_targets(
        upazilas=list(args.upazilas),
        max_per_class_per_upazila=int(args.max_per_class_per_upazila),
        val_frac=float(args.val_frac),
        min_val_per_class=int(args.min_val_per_class),
        min_val_frac_per_class=float(args.min_val_frac_per_class),
    )
    train_rem = train_targets.copy()
    val_rem = val_targets.copy()

    block_px = max(1, int(round(int(args.block_size_m) / 10.0)))
    per_upazila_stats: Dict[str, Dict[str, int]] = {}

    all_Xtr: List[np.ndarray] = []
    all_ytr: List[np.ndarray] = []
    all_Xva: List[np.ndarray] = []
    all_yva: List[np.ndarray] = []

    with rasterio.open(args.ae) as ae_src:
        if ae_src.count != 64:
            raise SystemExit(f"Expected 64 bands, found {ae_src.count} in {args.ae}")
        if ae_src.crs is None:
            raise SystemExit("AlphaEarth raster has no CRS.")

        up_list = list(args.upazilas)
        rng_up = np.random.default_rng(int(args.seed))
        rng_up.shuffle(up_list)

        log(f"Upazila order (seeded shuffle): {up_list}")
        log(f"labels-dir: {args.labels_dir}")
        log(f"Per-upazila per-class budget: {args.max_per_class_per_upazila}")
        log(f"Val targets per class: {val_targets}")
        log(f"Train targets per class: {train_targets}")
        log(f"Erosion radius for TRAIN only: {args.erode_px}px")
        log(f"AE validity requires >= {args.min_nonzero_bands} non-nodata/nonzero bands")

        for upazila in up_list:
            # Early stop if global targets are satisfied
            if all(train_rem[c] <= 0 and val_rem[c] <= 0 for c in range(1, 11)):
                log("All global targets satisfied. Stopping early.")
                break

            label_path = label_paths[upazila]
            log(f"Extracting from {upazila}: {label_path}")

            with rasterio.open(label_path) as lbl:
                if lbl.crs != ae_src.crs:
                    raise SystemExit(f"[{upazila}] CRS mismatch: label {lbl.crs} vs AE {ae_src.crs}")

                upazila_remaining = {c: int(args.max_per_class_per_upazila) for c in range(1, 11)}

                Xtr_parts, ytr_parts, Xva_parts, yva_parts, stats = extract_from_upazila(
                    ae_src=ae_src,
                    lbl=lbl,
                    upazila=upazila,
                    label_nodata=int(args.label_nodata),
                    ae_nodata=float(args.ae_nodata),
                    min_nonzero_bands=int(args.min_nonzero_bands),
                    chunk=int(args.chunk),
                    erode_r=int(args.erode_px),
                    block_px=int(block_px),
                    seed=int(args.seed),
                    val_frac_base=float(args.val_frac),
                    upazila_remaining=upazila_remaining,
                    train_rem=train_rem,
                    val_rem=val_rem,
                )

                if Xtr_parts:
                    all_Xtr.append(np.concatenate(Xtr_parts, axis=0))
                    all_ytr.append(np.concatenate(ytr_parts, axis=0))
                if Xva_parts:
                    all_Xva.append(np.concatenate(Xva_parts, axis=0))
                    all_yva.append(np.concatenate(yva_parts, axis=0))

                per_upazila_stats[upazila] = stats
                log(f"{upazila} samples -> train={stats['train']} val={stats['val']}")
                log(f"Remaining global train_rem: {train_rem}")
                log(f"Remaining global val_rem  : {val_rem}")

    if not all_Xtr or not all_ytr:
        raise SystemExit("No training samples extracted. Check overlap / nodata / labels-dir / erosion / min_nonzero_bands.")

    X_train = np.concatenate(all_Xtr, axis=0).astype(np.float32)
    y_train = np.concatenate(all_ytr, axis=0).astype(np.uint8)

    X_val = np.concatenate(all_Xva, axis=0).astype(np.float32) if all_Xva else np.zeros((0, 64), np.float32)
    y_val = np.concatenate(all_yva, axis=0).astype(np.uint8) if all_yva else np.zeros((0,), np.uint8)

    meta = {
        "ae_path": str(args.ae),
        "labels_dir": str(args.labels_dir),
        "upazilas": up_list,
        "label_paths": {u: str(label_paths[u]) for u in up_list},
        "max_per_class_per_upazila": int(args.max_per_class_per_upazila),
        "val_frac_base": float(args.val_frac),
        "min_val_per_class": int(args.min_val_per_class),
        "min_val_frac_per_class": float(args.min_val_frac_per_class),
        "block_size_m": int(args.block_size_m),
        "block_px": int(block_px),
        "seed": int(args.seed),
        "erode_px_train_only": int(args.erode_px),
        "min_nonzero_bands": int(args.min_nonzero_bands),
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
