"""
Mosaic Sentinel-2 L2A band tiles into a single GeoTIFF
using GDAL VRT + single gdalwarp -tap (pixel-aligned).

Usage:
  python scripts/sentinel/mosaic_sentinel_tiles.py --year 2017 --band B02 --resolution 10
"""

from __future__ import annotations

import argparse
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
    default_base = Path("/media/abdul-aziz/345E19F75E19B29A/bd_coastal_tiles")

    p = argparse.ArgumentParser(
        description="Mosaic Sentinel-2 JP2 tiles using GDAL VRT + gdalwarp -tap"
    )
    p.add_argument("--base-dir", type=Path, default=default_base)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--band", default="B02")
    p.add_argument("--resolution", type=int, choices=(10, 20, 60), default=10)
    p.add_argument("--target-crs", default="EPSG:32646")
    p.add_argument("--resampling", default="near", choices=("near", "bilinear", "cubic"))
    p.add_argument("--gdal-cache-mb", type=int, default=1024)
    p.add_argument("--output", type=Path, default=None)

    return p.parse_args(argv)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def list_band_paths(base_dir: Path, year: int, band: str, resolution: int) -> list[Path]:
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(year_dir)

    pat = f"*_{band}_{resolution}m.jp2"
    paths = sorted(year_dir.glob(f"**/{pat}"))
    if not paths:
        raise RuntimeError(f"No tiles found for {pat}")

    return paths


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

    tile_paths = list_band_paths(
        args.base_dir, args.year, args.band, args.resolution
    )

    log(
        f"Found {len(tile_paths)} tiles "
        f"for band {args.band} ({args.resolution} m)"
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
