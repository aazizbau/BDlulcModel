"""
Mosaic Sentinel-2 L2A band tiles into a single GeoTIFF
using GDAL VRT + single gdalwarp -tap (pixel-aligned).

Usage:
  python scripts/sentinel/mosaic_sentinel_tiles.py --year 2017 --band B02 --source-resolution 10 --resolution 10
  python scripts/sentinel/mosaic_sentinel_tiles.py --year 2017 --band B11 --source-resolution 20 --resolution 10

Reproduction and AOI adaptation
-------------------------------
Workflow role: Mosaic Sentinel-2 tiles and harmonize their projection, resolution, and extent.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--base-dir``, ``--year``, ``--band``, ``--source-resolution``, ``--resolution``, ``--target-crs``, ``--resampling``, ``--gdal-cache-mb``, ``--output``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace tile directories and AOI/grid settings; choose bilinear/cubic processing for continuous reflectance and never use it for categorical labels.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_base = Path(os.environ.get("BD_COASTAL_TILES_DIR", "data/raw/bd_coastal_tiles"))

    p = argparse.ArgumentParser(
        description="Mosaic Sentinel-2 JP2 tiles using GDAL VRT + gdalwarp -tap"
    )
    p.add_argument("--base-dir", type=Path, default=default_base)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--band", default="B02")
    p.add_argument(
        "--source-resolution",
        type=int,
        choices=(10, 20, 60),
        default=None,
        help="Resolution of source JP2 tiles. If omitted, infer from --resolution or band.",
    )
    p.add_argument("--resolution", type=int, choices=(10, 20, 60), default=10)
    p.add_argument("--target-crs", default="EPSG:32646")
    p.add_argument("--resampling", default="near", choices=("near", "bilinear", "cubic"))
    p.add_argument("--gdal-cache-mb", type=int, default=1024)
    p.add_argument("--output", type=Path, default=None)

    return p.parse_args(argv)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def list_band_paths(
    base_dir: Path,
    year: int,
    band: str,
    target_resolution: int,
    source_resolution: int | None,
) -> tuple[list[Path], int]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(year_dir)

    band_res_map = {
        "B01": 60,
        "B02": 10,
        "B03": 10,
        "B04": 10,
        "B05": 20,
        "B06": 20,
        "B07": 20,
        "B08": 10,
        "B8A": 20,
        "B09": 60,
        "B10": 60,
        "B11": 20,
        "B12": 20,
    }

    def _find(resolution: int) -> list[Path]:
        pat = f"*_{band}_{resolution}m.jp2"
        return sorted(year_dir.glob(f"**/{pat}"))

    if source_resolution is not None:
        paths = _find(source_resolution)
        resolved_source = source_resolution
    else:
        paths = _find(target_resolution)
        resolved_source = target_resolution
        if not paths:
            fallback = band_res_map.get(band)
            if fallback is not None and fallback != target_resolution:
                paths = _find(fallback)
                resolved_source = fallback

    if not paths:
        pat = f"*_{band}_{target_resolution}m.jp2"
        raise RuntimeError(f"No tiles found for {pat}")

    return paths, resolved_source


def default_output(base_dir: Path, year: int, band: str, res: int) -> Path:
    return base_dir / str(year) / f"coastal_{year}_{res:02d}_{band}.tif"


# ---------------------------------------------------------------------
# GDAL pipeline
# ---------------------------------------------------------------------
def run_gdal_pipeline(
    tile_paths: list[Path],
    out_tif: Path,
    *,
    target_crs: str,
    resolution: int,
    resampling: str,
    gdal_cache_mb: int,
) -> None:
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_tif.parent / "_tmp_reprojected"
    tmp_dir.mkdir(exist_ok=True)

    warped_tiles = []

    log("Reprojecting tiles to common CRS …")

    for p in tile_paths:
        warped = tmp_dir / (p.stem + "_warp.tif")

        cmd = [
            "gdalwarp",
            "-t_srs", target_crs,
            "-tr", str(resolution), str(resolution),
            "-tap",
            "-r", resampling,
            "-srcnodata", "0",
            "-dstnodata", "0",
            "-co", "TILED=YES",
            "-co", "COMPRESS=DEFLATE",
            "-overwrite",
            str(p),
            str(warped),
        ]

        subprocess.run(cmd, check=True)
        warped_tiles.append(warped)

    vrt_path = out_tif.with_suffix(".vrt")

    log("Building VRT from aligned tiles …")
    cmd_vrt = [
        "gdalbuildvrt",
        "-srcnodata", "0",
        "-vrtnodata", "0",
        str(vrt_path),
        *map(str, warped_tiles),
    ]
    subprocess.run(cmd_vrt, check=True)

    log("Translating VRT → final GeoTIFF …")
    cmd_translate = [
        "gdal_translate",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        str(vrt_path),
        str(out_tif),
    ]
    subprocess.run(cmd_translate, check=True)

    log(f"Saved mosaic → {out_tif}")
    # Treat DN=1 seam pixels as NoData (Sentinel-2 L2A edge fill)
    subprocess.run(
        ["gdal_edit.py", "-a_nodata", "1", str(out_tif)],
        check=True
    )
    # ------------------------------------------------------------
    # 2) Fill seam NoData pixels from nearest valid neighbors
    #    (NO interpolation, NO smoothing)
    # ------------------------------------------------------------
    filled_tif = out_tif.with_name(out_tif.stem + "_filled.tif")

    subprocess.run(
        [
            "gdal_fillnodata.py",
            "-md", "1",    # max distance = 1 pixel (only seams)
            "-si", "0",    # no smoothing
            str(out_tif),
            str(filled_tif),
        ],
        check=True
    )

    # ------------------------------------------------------------
    # 3) Replace original mosaic with filled version
    # ------------------------------------------------------------
    out_tif.unlink()
    filled_tif.rename(out_tif)

    log("Seam pixels filled from neighboring tiles (nearest neighbor)")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    out_tif = args.output or default_output(
        args.base_dir, args.year, args.band, args.resolution
    )

    tile_paths, source_resolution = list_band_paths(
        args.base_dir,
        args.year,
        args.band,
        args.resolution,
        args.source_resolution,
    )

    log(
        f"Found {len(tile_paths)} tiles for band {args.band} "
        f"({source_resolution} m → {args.resolution} m)"
    )

    run_gdal_pipeline(
        tile_paths,
        out_tif,
        target_crs=args.target_crs,
        resolution=args.resolution,
        resampling=args.resampling,
        gdal_cache_mb=args.gdal_cache_mb,
    )


if __name__ == "__main__":
    main()
