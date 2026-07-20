#!/usr/bin/env python3
"""
Align 10 full-coast spectral index rasters to a yearly AlphaEarth inference grid.

Default behavior for a given year:
- AE reference:
    data/interim/bd_coastal_alphaearth_<year>_utm46_f32.tif
- Input index rasters:
    data/interim/bdcoastal_solid_<year>_utm46_<index>.tif
- Output folder:
    data/interim/ae_aligned_indices_<year>_fullcoast/
- Output filename example:
    bdcoastal_solid_<year>_utm46_ndvi_aegrid.tif

Example:
python scripts/inference/make_align_10indices_with_AEGrid.py \
    --year 2017

Explicit output folder:
python scripts/inference/make_align_10indices_with_AEGrid.py \
    --year 2024 \
    --outdir data/interim/ae_aligned_indices_2024_fullcoast

Reproduction and AOI adaptation
-------------------------------
Workflow role: Prepare inference features, apply the selected classifier, or derive classified-map change products.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--year``, ``--ae``, ``--interim-dir``, ``--outdir``, ``--src-nodata``, ``--dst-nodata``, ``--resampling``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace feature rasters and AOI paths, but keep every feature on the exact training grid and use a checkpoint trained with the same feature order and class mapping.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
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
    print(f"[{ts}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Align 10 full-coast index rasters to the exact yearly AE inference grid."
    )
    p.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process, e.g. 2017 or 2024.",
    )
    p.add_argument(
        "--ae",
        type=Path,
        default=None,
        help="Optional yearly AE reference raster. Default: data/interim/bd_coastal_alphaearth_<year>_utm46_f32.tif",
    )
    p.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory containing the 10 input index rasters.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for AE-grid-aligned indices. Default: data/interim/ae_aligned_indices_<year>_fullcoast",
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


def default_ae_path(year: int) -> Path:
    return Path(f"data/interim/bd_coastal_alphaearth_{year}_utm46_f32.tif")


def default_outdir(year: int) -> Path:
    return Path(f"data/interim/ae_aligned_indices_{year}_fullcoast")


def build_input_paths(interim_dir: Path, year: int) -> Dict[str, Path]:
    return {
        name: interim_dir / f"bdcoastal_solid_{year}_utm46_{name}.tif"
        for name in INDEX_ORDER
    }


def build_output_path(outdir: Path, year: int, index_name: str) -> Path:
    return outdir / f"bdcoastal_solid_{year}_utm46_{index_name}_aegrid.tif"


def get_resampling(name: str) -> Resampling:
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    return mapping[name]


def same_grid(
    src_a: rasterio.DatasetReader,
    src_b: rasterio.DatasetReader,
    atol: float = 1e-9,
) -> bool:
    if src_a.crs != src_b.crs:
        return False
    if src_a.width != src_b.width or src_a.height != src_b.height:
        return False
    a = src_a.transform
    b = src_b.transform
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
            SCRIPT_NAME="make_align_10indices_with_AEGrid.py",
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
    args.ae = resolve_path(args.ae or default_ae_path(args.year))
    args.interim_dir = resolve_path(args.interim_dir)
    args.outdir = resolve_path(args.outdir or default_outdir(args.year))

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
                if same_grid(src, ae_src):
                    log("Grid match      : already matches AE grid")
                else:
                    log("Grid match      : reprojection/resampling required")

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
