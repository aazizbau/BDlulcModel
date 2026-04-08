#!/usr/bin/env python3
"""
Run full-coast blockwise inference with the best AE64 + 10 indices MLP model.

For a given year, this script will:
- load the best model checkpoint
- load training normalization stats from the 2023 train/val/test NPZ
- read the yearly AE raster and aligned index rasters
- stack features in the locked order:
    ae_01..ae_64, ndvi, evi, msavi, ndmi, ndwi, ndpi, ndbi, bsi, nirv, awei_sh
- normalize features using the 2023 training mu/sigma
- run blockwise inference
- write:
    * class raster
    * confidence raster
    * uncertainty raster (1 - max probability)
    * optional probability stack
- generate analysis-ready CSVs

Default yearly inputs:
- AE:
    data/interim/bd_coastal_alphaearth_<year>_utm46_f32.tif
- aligned indices dir:
    data/interim/ae_aligned_indices_<year>_fullcoast
- checkpoint:
    runs/mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3/best_model.pt
- training NPZ:
    data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz

Default outputs for year 2017:
- outputs/inference/2017/lulc_class_2017.tif
- outputs/inference/2017/confidence_2017.tif
- outputs/inference/2017/uncertainty_2017.tif
- outputs/inference/2017/inference_summary_2017.csv
- outputs/inference/2017/class_area_summary_2017.csv
- outputs/inference/2017/confidence_bin_summary_2017.csv
- outputs/inference/2017/raster_inventory_2017.csv
- outputs/inference/2017/locked_feature_order_2017.csv

Example run:
python scripts/inference/make_inference_with_bestmodel.py \
  --year 2017

python scripts/inference/make_inference_with_bestmodel.py \
  --year 2024 \
  --save-probability-stack # for optional probability stack

"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = Path("runs/mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3/best_model.pt")
DEFAULT_TRAIN_NPZ = Path("data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz")
DEFAULT_OUTPUT_ROOT = Path("outputs/inference")

AE_NODATA = 0.0
INDEX_NODATA = -9999.0
RASTER_FLOAT_NODATA = -9999.0
NUM_CLASSES_FIXED = 10
EXPECTED_INPUT_DIM = 74

INDEX_ORDER = [
    "ndvi",
    "evi",
    "msavi",
    "ndmi",
    "ndwi",
    "ndpi",
    "ndbi",
    "bsi",
    "nirv",
    "awei_sh",
]
FEATURE_ORDER = [f"ae_{i:02d}" for i in range(1, 65)] + INDEX_ORDER


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_ae_path(year: int) -> Path:
    return Path(f"data/interim/bd_coastal_alphaearth_{year}_utm46_f32.tif")


def default_indices_dir(year: int) -> Path:
    return Path(f"data/interim/ae_aligned_indices_{year}_fullcoast")


def default_index_path(year: int, index_name: str, indices_dir: Path) -> Path:
    return indices_dir / f"bdcoastal_solid_{year}_utm46_{index_name}_aegrid.tif"


def default_year_output_dir(year: int, output_root: Path) -> Path:
    return output_root / str(year)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run blockwise full-raster inference with the best AE64 + 10 indices MLP model."
    )
    p.add_argument("--year", type=int, required=True, help="Target year, e.g. 2017 or 2024.")
    p.add_argument("--ae", type=Path, default=None, help="Optional AE raster override.")
    p.add_argument("--indices-dir", type=Path, default=None, help="Optional aligned indices directory override.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Best model checkpoint.")
    p.add_argument("--train-npz", type=Path, default=DEFAULT_TRAIN_NPZ, help="Training NPZ with mu/sigma and feature names.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root directory.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"], help="Inference device.")
    p.add_argument("--batch-size", type=int, default=8192, help="Batch size for model forward passes.")
    p.add_argument("--save-probability-stack", action="store_true", help="Write a 10-band probability stack GeoTIFF.")
    return p.parse_args()


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RasterPaths:
    class_tif: Path
    confidence_tif: Path
    uncertainty_tif: Path
    probability_tif: Path | None


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MLPClassifier(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        num_classes=int(ckpt["num_classes"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_train_stats(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    with np.load(npz_path, allow_pickle=True) as d:
        required = ["mu", "sigma", "feature_names", "meta"]
        missing = [k for k in required if k not in d]
        if missing:
            raise SystemExit(f"Training NPZ missing required keys: {missing}")
        mu = d["mu"].astype(np.float32)
        sigma = d["sigma"].astype(np.float32)
        feature_names = [str(x) for x in d["feature_names"].tolist()]
        meta_raw = d["meta"]
        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.item()
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        if isinstance(meta_raw, str):
            meta = json.loads(meta_raw)
        elif isinstance(meta_raw, dict):
            meta = meta_raw
        else:
            meta = {}
    return mu, sigma, feature_names, meta


def same_grid(src_a: rasterio.DatasetReader, src_b: rasterio.DatasetReader, tol: float = 1e-9) -> bool:
    if src_a.crs != src_b.crs:
        return False
    if src_a.width != src_b.width or src_a.height != src_b.height:
        return False
    a = src_a.transform
    b = src_b.transform
    return (
        abs(a.a - b.a) <= tol
        and abs(a.b - b.b) <= tol
        and abs(a.c - b.c) <= tol
        and abs(a.d - b.d) <= tol
        and abs(a.e - b.e) <= tol
        and abs(a.f - b.f) <= tol
    )


class IndexStack:
    def __init__(self, year: int, indices_dir: Path) -> None:
        self.year = year
        self.indices_dir = indices_dir
        self.datasets: Dict[str, rasterio.DatasetReader] = {}

    def __enter__(self) -> "IndexStack":
        for name in INDEX_ORDER:
            path = default_index_path(self.year, name, self.indices_dir)
            self.datasets[name] = rasterio.open(path)
        return self

    def read(self, name: str, window: Window) -> np.ndarray:
        return self.datasets[name].read(1, window=window).astype(np.float32)

    def __exit__(self, exc_type, exc, tb) -> None:
        for ds in self.datasets.values():
            ds.close()
        self.datasets.clear()


def validate_inputs(ae_path: Path, indices_dir: Path, year: int) -> None:
    if not ae_path.exists():
        raise SystemExit(f"AE raster not found: {ae_path}")
    missing = [str(default_index_path(year, n, indices_dir)) for n in INDEX_ORDER if not default_index_path(year, n, indices_dir).exists()]
    if missing:
        raise SystemExit("Missing aligned index rasters:\n" + "\n".join(missing))

    with rasterio.open(ae_path) as ae:
        if ae.count != 64:
            raise SystemExit(f"Expected 64 AE bands, found {ae.count}: {ae_path}")
        if ae.nodata is None or float(ae.nodata) != AE_NODATA:
            raise SystemExit(f"AE nodata mismatch. Expected {AE_NODATA}, found {ae.nodata}: {ae_path}")
        for name in INDEX_ORDER:
            path = default_index_path(year, name, indices_dir)
            with rasterio.open(path) as idx:
                if idx.count != 1:
                    raise SystemExit(f"Index raster must be single-band: {path}")
                if idx.nodata is None or float(idx.nodata) != INDEX_NODATA:
                    raise SystemExit(f"Index nodata mismatch. Expected {INDEX_NODATA}, found {idx.nodata}: {path}")
                if not same_grid(ae, idx):
                    raise SystemExit(f"Grid mismatch between AE and {name}: {path}")


def normalize_block(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma_safe = np.where(sigma == 0, 1.0, sigma).astype(np.float32)
    return ((X - mu) / sigma_safe).astype(np.float32)


def make_output_profiles(ae_src: rasterio.DatasetReader, save_probability_stack: bool) -> Tuple[dict, dict, dict, dict | None]:
    class_profile = ae_src.profile.copy()
    class_profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "ZSTD",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "IF_SAFER",
        }
    )

    float_profile = ae_src.profile.copy()
    float_profile.update(
        {
            "count": 1,
            "dtype": "float32",
            "nodata": RASTER_FLOAT_NODATA,
            "compress": "ZSTD",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "IF_SAFER",
        }
    )

    prob_profile = None
    if save_probability_stack:
        prob_profile = ae_src.profile.copy()
        prob_profile.update(
            {
                "count": NUM_CLASSES_FIXED,
                "dtype": "float32",
                "nodata": RASTER_FLOAT_NODATA,
                "compress": "ZSTD",
                "predictor": 3,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "interleave": "band",
                "BIGTIFF": "IF_SAFER",
            }
        )

    return class_profile, float_profile, float_profile.copy(), prob_profile


def build_raster_paths(year_dir: Path, year: int, save_probability_stack: bool) -> RasterPaths:
    probability_tif = year_dir / f"probability_stack_{year}.tif" if save_probability_stack else None
    return RasterPaths(
        class_tif=year_dir / f"lulc_class_{year}.tif",
        confidence_tif=year_dir / f"confidence_{year}.tif",
        uncertainty_tif=year_dir / f"uncertainty_{year}.tif",
        probability_tif=probability_tif,
    )


def write_feature_order_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_position_1based", "feature_name"])
        for i, name in enumerate(FEATURE_ORDER, start=1):
            writer.writerow([i, name])


def write_inventory_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_class_area_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "class_id",
                "pixel_count",
                "area_m2",
                "area_km2",
                "mean_confidence",
                "mean_uncertainty",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_confidence_bins_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "bin_lower",
                "bin_upper",
                "pixel_count",
                "area_m2",
                "area_km2",
                "fraction_of_valid_pixels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, row: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def iter_output_paths(paths: RasterPaths) -> Iterable[Path]:
    yield paths.class_tif
    yield paths.confidence_tif
    yield paths.uncertainty_tif
    if paths.probability_tif is not None:
        yield paths.probability_tif


def main() -> None:
    args = parse_args()

    ae_path = resolve_path(args.ae or default_ae_path(args.year))
    indices_dir = resolve_path(args.indices_dir or default_indices_dir(args.year))
    checkpoint_path = resolve_path(args.checkpoint)
    train_npz_path = resolve_path(args.train_npz)
    output_root = resolve_path(args.output_root)
    year_dir = default_year_output_dir(args.year, output_root)
    year_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not train_npz_path.exists():
        raise SystemExit(f"Training NPZ not found: {train_npz_path}")

    validate_inputs(ae_path, indices_dir, args.year)

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(checkpoint_path, device)
    mu_npz, sigma_npz, feature_names_npz, npz_meta = load_train_stats(train_npz_path)

    feature_names_ckpt = [str(x) for x in ckpt["feature_names"]]
    if feature_names_ckpt != FEATURE_ORDER:
        raise SystemExit("Checkpoint feature order does not match expected locked feature order.")
    if feature_names_npz != FEATURE_ORDER:
        raise SystemExit("Training NPZ feature order does not match expected locked feature order.")

    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sigma = np.asarray(ckpt["sigma"], dtype=np.float32)
    if mu.shape[0] != EXPECTED_INPUT_DIM or sigma.shape[0] != EXPECTED_INPUT_DIM:
        raise SystemExit("Checkpoint mu/sigma do not have expected length 74.")
    if not np.allclose(mu, mu_npz) or not np.allclose(sigma, sigma_npz):
        raise SystemExit("Checkpoint mu/sigma do not match training NPZ mu/sigma.")

    raster_paths = build_raster_paths(year_dir, args.year, args.save_probability_stack)
    inventory_rows: List[dict] = []

    log(f"Year            : {args.year}")
    log(f"AE raster       : {ae_path}")
    log(f"Indices dir     : {indices_dir}")
    log(f"Checkpoint      : {checkpoint_path}")
    log(f"Training NPZ    : {train_npz_path}")
    log(f"Device          : {device}")
    log(f"Batch size      : {args.batch_size}")
    log(f"Save prob stack : {args.save_probability_stack}")

    for name in INDEX_ORDER:
        inventory_rows.append({"role": "aligned_index", "name": name, "path": str(default_index_path(args.year, name, indices_dir))})
    inventory_rows.extend(
        [
            {"role": "ae_raster", "name": f"ae_{args.year}", "path": str(ae_path)},
            {"role": "checkpoint", "name": "best_model", "path": str(checkpoint_path)},
            {"role": "train_npz", "name": "train_stats", "path": str(train_npz_path)},
        ]
    )

    class_counts = np.zeros(NUM_CLASSES_FIXED + 1, dtype=np.int64)
    class_conf_sums = np.zeros(NUM_CLASSES_FIXED + 1, dtype=np.float64)
    class_unc_sums = np.zeros(NUM_CLASSES_FIXED + 1, dtype=np.float64)
    conf_bin_edges = np.linspace(0.0, 1.0, 11)
    conf_bin_counts = np.zeros(10, dtype=np.int64)
    valid_pixels_total = 0

    start = datetime.now(JST)

    with rasterio.open(ae_path) as ae_src, IndexStack(args.year, indices_dir) as idx_stack:
        class_profile, conf_profile, unc_profile, prob_profile = make_output_profiles(ae_src, args.save_probability_stack)

        with rasterio.open(raster_paths.class_tif, "w", **class_profile) as class_dst, \
            rasterio.open(raster_paths.confidence_tif, "w", **conf_profile) as conf_dst, \
            rasterio.open(raster_paths.uncertainty_tif, "w", **unc_profile) as unc_dst:

            prob_dst = rasterio.open(raster_paths.probability_tif, "w", **prob_profile) if raster_paths.probability_tif and prob_profile else None
            try:
                total_windows = sum(1 for _ in ae_src.block_windows(1))
                processed_windows = 0

                for _, window in ae_src.block_windows(1):
                    processed_windows += 1
                    ae_block = ae_src.read(list(range(1, ae_src.count + 1)), window=window).astype(np.float32)
                    h, w = ae_block.shape[1], ae_block.shape[2]

                    class_out = np.zeros((h, w), dtype=np.uint8)
                    conf_out = np.full((h, w), RASTER_FLOAT_NODATA, dtype=np.float32)
                    unc_out = np.full((h, w), RASTER_FLOAT_NODATA, dtype=np.float32)
                    prob_out = np.full((NUM_CLASSES_FIXED, h, w), RASTER_FLOAT_NODATA, dtype=np.float32) if prob_dst is not None else None

                    if AE_NODATA == 0.0:
                        ae_valid = np.all(ae_block != 0.0, axis=0)
                    else:
                        ae_valid = np.all(ae_block != float(AE_NODATA), axis=0)
                    ae_valid &= np.all(np.isfinite(ae_block), axis=0)

                    idx_arrays = []
                    idx_valid = np.ones((h, w), dtype=bool)
                    for name in INDEX_ORDER:
                        arr = idx_stack.read(name, window=window)
                        idx_arrays.append(arr)
                        idx_valid &= np.isfinite(arr) & (arr != float(INDEX_NODATA))

                    valid_mask = ae_valid & idx_valid
                    if np.any(valid_mask):
                        rr, cc = np.where(valid_mask)
                        X_ae = ae_block[:, rr, cc].T
                        X_idx = np.stack([arr[rr, cc] for arr in idx_arrays], axis=1).astype(np.float32)
                        X = np.concatenate([X_ae, X_idx], axis=1).astype(np.float32)

                        if X.shape[1] != EXPECTED_INPUT_DIM:
                            raise SystemExit(f"Expected 74 features in block, found {X.shape[1]}")
                        if not np.isfinite(X).all():
                            raise SystemExit("Non-finite values found in valid inference features.")

                        X = normalize_block(X, mu, sigma)

                        probs_chunks: List[np.ndarray] = []
                        with torch.no_grad():
                            for start_idx in range(0, X.shape[0], args.batch_size):
                                xb = torch.from_numpy(X[start_idx:start_idx + args.batch_size]).to(device)
                                logits = model(xb)
                                probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
                                probs_chunks.append(probs)
                        probs_all = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, NUM_CLASSES_FIXED), dtype=np.float32)

                        pred_zero = np.argmax(probs_all, axis=1)
                        pred = (pred_zero + 1).astype(np.uint8)
                        conf = np.max(probs_all, axis=1).astype(np.float32)
                        unc = (1.0 - conf).astype(np.float32)

                        class_out[rr, cc] = pred
                        conf_out[rr, cc] = conf
                        unc_out[rr, cc] = unc
                        if prob_out is not None:
                            for band_idx in range(NUM_CLASSES_FIXED):
                                prob_out[band_idx, rr, cc] = probs_all[:, band_idx]

                        valid_pixels_total += pred.shape[0]
                        binc = np.bincount(pred, minlength=NUM_CLASSES_FIXED + 1)
                        class_counts[: len(binc)] += binc
                        for cls in range(1, NUM_CLASSES_FIXED + 1):
                            mask_cls = pred == cls
                            if np.any(mask_cls):
                                class_conf_sums[cls] += float(conf[mask_cls].sum())
                                class_unc_sums[cls] += float(unc[mask_cls].sum())
                        bin_ids = np.clip(np.digitize(conf, conf_bin_edges[1:], right=False), 0, 9)
                        for bi in range(10):
                            conf_bin_counts[bi] += int(np.sum(bin_ids == bi))

                    class_dst.write(class_out, 1, window=window)
                    conf_dst.write(conf_out, 1, window=window)
                    unc_dst.write(unc_out, 1, window=window)
                    if prob_dst is not None and prob_out is not None:
                        prob_dst.write(prob_out, window=window)

                    if processed_windows % 250 == 0 or processed_windows == total_windows:
                        log(f"Processed windows: {processed_windows}/{total_windows}")
            finally:
                if prob_dst is not None:
                    prob_dst.close()

        pixel_area_m2 = abs(ae_src.transform.a * ae_src.transform.e)
        inventory_rows.extend(
            [
                {"role": "output_raster", "name": "class_raster", "path": str(raster_paths.class_tif)},
                {"role": "output_raster", "name": "confidence_raster", "path": str(raster_paths.confidence_tif)},
                {"role": "output_raster", "name": "uncertainty_raster", "path": str(raster_paths.uncertainty_tif)},
            ]
        )
        if raster_paths.probability_tif is not None:
            inventory_rows.append({"role": "output_raster", "name": "probability_stack", "path": str(raster_paths.probability_tif)})

    end = datetime.now(JST)
    elapsed_seconds = (end - start).total_seconds()

    feature_order_csv = year_dir / f"locked_feature_order_{args.year}.csv"
    inventory_csv = year_dir / f"raster_inventory_{args.year}.csv"
    class_area_csv = year_dir / f"class_area_summary_{args.year}.csv"
    conf_bins_csv = year_dir / f"confidence_bin_summary_{args.year}.csv"
    summary_csv = year_dir / f"inference_summary_{args.year}.csv"

    write_feature_order_csv(feature_order_csv)
    write_inventory_csv(inventory_csv, inventory_rows)

    class_rows: List[dict] = []
    for cls in range(1, NUM_CLASSES_FIXED + 1):
        pixels = int(class_counts[cls])
        area_m2 = pixels * pixel_area_m2
        class_rows.append(
            {
                "year": args.year,
                "class_id": cls,
                "pixel_count": pixels,
                "area_m2": area_m2,
                "area_km2": area_m2 / 1_000_000.0,
                "mean_confidence": (class_conf_sums[cls] / pixels) if pixels > 0 else "",
                "mean_uncertainty": (class_unc_sums[cls] / pixels) if pixels > 0 else "",
            }
        )
    write_class_area_csv(class_area_csv, class_rows)

    bin_rows: List[dict] = []
    for i in range(10):
        lower = conf_bin_edges[i]
        upper = conf_bin_edges[i + 1]
        pixels = int(conf_bin_counts[i])
        area_m2 = pixels * pixel_area_m2
        bin_rows.append(
            {
                "year": args.year,
                "bin_lower": lower,
                "bin_upper": upper,
                "pixel_count": pixels,
                "area_m2": area_m2,
                "area_km2": area_m2 / 1_000_000.0,
                "fraction_of_valid_pixels": (pixels / valid_pixels_total) if valid_pixels_total > 0 else "",
            }
        )
    write_confidence_bins_csv(conf_bins_csv, bin_rows)

    summary_row = {
        "year": args.year,
        "started_at_jst": start.isoformat(timespec="seconds"),
        "finished_at_jst": end.isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "batch_size": args.batch_size,
        "checkpoint_path": str(checkpoint_path),
        "train_npz_path": str(train_npz_path),
        "ae_raster": str(ae_path),
        "indices_dir": str(indices_dir),
        "valid_pixels": int(valid_pixels_total),
        "pixel_area_m2": pixel_area_m2,
        "class_raster": str(raster_paths.class_tif),
        "confidence_raster": str(raster_paths.confidence_tif),
        "uncertainty_raster": str(raster_paths.uncertainty_tif),
        "probability_stack": str(raster_paths.probability_tif) if raster_paths.probability_tif is not None else "",
        "feature_order_csv": str(feature_order_csv),
        "inventory_csv": str(inventory_csv),
        "class_area_csv": str(class_area_csv),
        "confidence_bin_csv": str(conf_bins_csv),
        "npz_year_meta": npz_meta.get("year", ""),
        "checkpoint_best_val_macro_f1": ckpt.get("best_val_macro_f1", ""),
        "checkpoint_epoch": ckpt.get("epoch", ""),
    }
    write_summary_csv(summary_csv, summary_row)

    log(f"Saved class raster       : {raster_paths.class_tif}")
    log(f"Saved confidence raster  : {raster_paths.confidence_tif}")
    log(f"Saved uncertainty raster : {raster_paths.uncertainty_tif}")
    if raster_paths.probability_tif is not None:
        log(f"Saved probability stack  : {raster_paths.probability_tif}")
    log(f"Saved summary CSV        : {summary_csv}")
    log(f"Saved class-area CSV     : {class_area_csv}")
    log(f"Saved confidence-bin CSV : {conf_bins_csv}")
    log(f"Saved inventory CSV      : {inventory_csv}")
    log(f"Saved feature-order CSV  : {feature_order_csv}")


if __name__ == "__main__":
    main()
