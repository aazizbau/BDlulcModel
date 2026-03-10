#!/usr/bin/env python3
"""
Align 10 spectral index rasters to an AlphaEarth reference grid.

This script:
- uses the AE raster as the exact target grid
- clips each input index raster to the AE extent
- resamples each input index raster to exactly the AE grid
- writes outputs with:
    * same CRS as AE
    * same width/height as AE
    * same transform as AE
    * Float32 dtype
    * nodata preserved as -9999

Default output folder:
  data/interim/ae_aligned_indices_2023

Default output filename pattern:
  bdcoastal_4upazila_2023_utm46_<index>_aegrid.tif

Example:
  python scripts/training/align_10indices_to_ae_grid.py \
    --ae data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif

Optional custom output dir:
  python scripts/training/align_10indices_to_ae_grid.py \
    --ae data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif \
    --outdir data/interim/ae_aligned_indices_2023
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_ORDER: List[str] = [
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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clip and align 10 spectral index rasters to the exact AE reference grid."
    )
    p.add_argument(
        "--ae",
        type=Path,
        required=True,
        help="Reference AlphaEarth raster (defines exact output grid).",
    )
    p.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year used in default input/output naming (default: 2023).",
    )
    p.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Folder containing original full-area index rasters (default: data/interim).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/interim/ae_aligned_indices_2023"),
        help="Output folder for AE-grid-aligned indices.",
    )
    p.add_argument(
        "--src-nodata",
        type=float,
        default=-9999.0,
        help="Input nodata value for source index rasters (default: -9999.0).",
    )
    p.add_argument(
        "--dst-nodata",
        type=float,
        default=-9999.0,
        help="Output nodata value (default: -9999.0).",
    )
    p.add_argument(
        "--resampling",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "cubic"],
        help="Resampling method for continuous indices (default: bilinear).",
    )
    return p.parse_args()


def build_input_paths(interim_dir: Path, year: int) -> Dict[str, Path]:
    return {
        name: interim_dir / f"bdcoastal_solid_{year}_utm46_{name}.tif"
        for name in INDEX_ORDER
    }


def build_output_path(outdir: Path, year: int, index_name: str) -> Path:
    return outdir / f"bdcoastal_4upazila_{year}_utm46_{index_name}_aegrid.tif"


def get_resampling(name: str) -> Resampling:
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    return mapping[name]


def same_grid(
    src_crs,
    src_transform,
    src_width: int,
    src_height: int,
    ref_crs,
    ref_transform,
    ref_width: int,
    ref_height: int,
    atol: float = 1e-9,
) -> bool:
    if src_crs != ref_crs:
        return False
    if src_width != ref_width or src_height != ref_height:
        return False

    a = src_transform
    b = ref_transform
    return (
        abs(a.a - b.a) <= atol
        and abs(a.b - b.b) <= atol
        and abs(a.c - b.c) <= atol
        and abs(a.d - b.d) <= atol
        and abs(a.e - b.e) <= atol
        and abs(a.f - b.f) <= atol
    )


def align_one_index(
    input_path: Path,
    output_path: Path,
    ref_profile: dict,
    src_nodata: float,
    dst_nodata: float,
    resampling: Resampling,
) -> None:
    with rasterio.open(input_path) as src:
        if src.count != 1:
            raise SystemExit(f"Expected single-band index raster, found {src.count}: {input_path}")

        if same_grid(
            src.crs,
            src.transform,
            src.width,
            src.height,
            ref_profile["crs"],
            ref_profile["transform"],
            ref_profile["width"],
            ref_profile["height"],
        ):
            log(f"Already matches AE grid, copying through reproject path: {input_path.name}")

        src_arr = src.read(1).astype(np.float32)

        dst_arr = np.full(
            (ref_profile["height"], ref_profile["width"]),
            dst_nodata,
            dtype=np.float32,
        )

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    out_profile = ref_profile.copy()
    out_profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "nodata": dst_nodata,
            "compress": "ZSTD",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(dst_arr, 1)
        dst.update_tags(
            AREA_OR_POINT="Area",
            SCRIPT_NAME="align_10indices_to_ae_grid.py",
            CREATED_AT_JST=datetime.now(JST).isoformat(timespec="seconds"),
            SOURCE_FILE=str(input_path),
            TARGET_AE_GRID="true",
            OUTPUT_NODATA=str(dst_nodata),
        )

    valid_mask = np.isfinite(dst_arr) & (dst_arr != dst_nodata)
    valid_count = int(valid_mask.sum())
    total_count = int(dst_arr.size)
    valid_fraction = valid_count / total_count if total_count else 0.0

    if valid_count > 0:
        vmin = float(dst_arr[valid_mask].min())
        vmax = float(dst_arr[valid_mask].max())
        vmean = float(dst_arr[valid_mask].mean())
    else:
        vmin = float("nan")
        vmax = float("nan")
        vmean = float("nan")

    log(f"Saved           : {output_path}")
    log(f"Valid pixels    : {valid_count}/{total_count} ({valid_fraction:.6f})")
    log(f"Value stats     : min={vmin:.6f} mean={vmean:.6f} max={vmax:.6f}")


def main() -> None:
    args = parse_args()
    args.ae = resolve_path(args.ae)
    args.interim_dir = resolve_path(args.interim_dir)
    args.outdir = resolve_path(args.outdir)

    if not args.ae.exists():
        raise SystemExit(f"AE reference raster not found: {args.ae}")

    input_paths = build_input_paths(args.interim_dir, args.year)
    missing = [str(p) for p in input_paths.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing input index rasters:\n" + "\n".join(missing))

    resampling = get_resampling(args.resampling)
    args.outdir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.ae) as ae_src:
        if ae_src.crs is None:
            raise SystemExit("AE reference raster has no CRS.")

        ref_profile = ae_src.profile.copy()
        ref_profile.update(
            {
                "crs": ae_src.crs,
                "transform": ae_src.transform,
                "width": ae_src.width,
                "height": ae_src.height,
            }
        )

        log(f"AE reference    : {args.ae}")
        log(f"AE CRS          : {ae_src.crs}")
        log(f"AE size         : {ae_src.width} x {ae_src.height}")
        log(f"AE transform    : {ae_src.transform}")
        log(f"Output dir      : {args.outdir}")
        log(f"Year            : {args.year}")
        log(f"Resampling      : {args.resampling}")
        log(f"Source nodata   : {args.src_nodata}")
        log(f"Output nodata   : {args.dst_nodata}")

        for idx_name in INDEX_ORDER:
            input_path = input_paths[idx_name]
            output_path = build_output_path(args.outdir, args.year, idx_name)

            log("-" * 72)
            log(f"Index           : {idx_name}")
            log(f"Input           : {input_path}")
            log(f"Output          : {output_path}")

            with rasterio.open(input_path) as src:
                log(f"Input CRS       : {src.crs}")
                log(f"Input size      : {src.width} x {src.height}")
                log(f"Input transform : {src.transform}")
                log(f"Input nodata    : {src.nodata}")

            align_one_index(
                input_path=input_path,
                output_path=output_path,
                ref_profile=ref_profile,
                src_nodata=args.src_nodata,
                dst_nodata=args.dst_nodata,
                resampling=resampling,
            )

    log("=" * 72)
    log("Finished aligning all 10 indices to AE grid.")
    log(f"Outputs saved in: {args.outdir}")


if __name__ == "__main__":
    main()
