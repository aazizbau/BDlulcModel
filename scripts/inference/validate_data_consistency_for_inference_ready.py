#!/usr/bin/env python3
"""
Validate yearly inference-ready AE + aligned-index data consistency.

Checks:
- AE has 64 bands
- every aligned index raster is single-band
- aligned index rasters match the AE raster in:
  * CRS
  * transform
  * width
  * height
- nodata is consistent:
  * AE: 0
  * indices: -9999
- locked feature order is reported and written to CSV

Default yearly inputs:
- AE:
    data/interim/bd_coastal_alphaearth_<year>_utm46_f32.tif
- aligned indices dir:
    data/interim/ae_aligned_indices_<year>_fullcoast/

Default CSV output:
- outputs/inference_ready_validation.csv

Example:
python scripts/inference/validate_data_consistency_for_inference_ready.py \
    --years 2017 2024

Reproduction and AOI adaptation
-------------------------------
Workflow role: Prepare inference features, apply the selected classifier, or derive classified-map change products.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--years``, ``--output-csv``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import rasterio


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("outputs/inference_ready_validation.csv")
AE_EXPECTED_BANDS = 64
AE_EXPECTED_NODATA = 0.0
INDEX_EXPECTED_BANDS = 1
INDEX_EXPECTED_NODATA = -9999.0
TRANSFORM_TOL = 1e-9

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate yearly AE + aligned-index inference-ready raster consistency."
    )
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2017, 2024],
        help="Years to validate (default: 2017 2024).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    return p.parse_args()


def default_ae_path(year: int) -> Path:
    return Path(f"data/interim/bd_coastal_alphaearth_{year}_utm46_f32.tif")


def default_aligned_dir(year: int) -> Path:
    return Path(f"data/interim/ae_aligned_indices_{year}_fullcoast")


def default_index_path(year: int, index_name: str) -> Path:
    return default_aligned_dir(year) / f"bdcoastal_solid_{year}_utm46_{index_name}_aegrid.tif"


def affine_matches(a: Any, b: Any, tol: float = TRANSFORM_TOL) -> bool:
    return (
        abs(a.a - b.a) <= tol
        and abs(a.b - b.b) <= tol
        and abs(a.c - b.c) <= tol
        and abs(a.d - b.d) <= tol
        and abs(a.e - b.e) <= tol
        and abs(a.f - b.f) <= tol
    )


def nodata_matches(actual: Any, expected: float, tol: float = 1e-12) -> bool:
    if actual is None:
        return False
    try:
        return abs(float(actual) - float(expected)) <= tol
    except Exception:
        return False


def validate_year(year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ae_path = resolve_path(default_ae_path(year))

    ae_exists = ae_path.exists()
    ae_meta: dict[str, Any] = {}

    if ae_exists:
        with rasterio.open(ae_path) as ae:
            ae_meta = {
                "count": ae.count,
                "crs": str(ae.crs),
                "transform": ae.transform,
                "transform_str": str(ae.transform),
                "width": ae.width,
                "height": ae.height,
                "nodata": ae.nodata,
            }
            rows.append(
                {
                    "year": year,
                    "record_type": "ae_reference",
                    "feature_position_1based": "",
                    "feature_name": "",
                    "path": str(ae_path),
                    "exists": True,
                    "band_count": ae.count,
                    "band_count_ok": ae.count == AE_EXPECTED_BANDS,
                    "crs": str(ae.crs),
                    "crs_match_ae": True,
                    "transform": str(ae.transform),
                    "transform_match_ae": True,
                    "width": ae.width,
                    "width_match_ae": True,
                    "height": ae.height,
                    "height_match_ae": True,
                    "nodata": ae.nodata,
                    "nodata_ok": nodata_matches(ae.nodata, AE_EXPECTED_NODATA),
                    "overall_ok": (ae.count == AE_EXPECTED_BANDS and nodata_matches(ae.nodata, AE_EXPECTED_NODATA)),
                    "notes": "",
                }
            )
    else:
        rows.append(
            {
                "year": year,
                "record_type": "ae_reference",
                "feature_position_1based": "",
                "feature_name": "",
                "path": str(ae_path),
                "exists": False,
                "band_count": "",
                "band_count_ok": False,
                "crs": "",
                "crs_match_ae": False,
                "transform": "",
                "transform_match_ae": False,
                "width": "",
                "width_match_ae": False,
                "height": "",
                "height_match_ae": False,
                "nodata": "",
                "nodata_ok": False,
                "overall_ok": False,
                "notes": "AE reference raster missing",
            }
        )

    for idx_pos, index_name in enumerate(INDEX_ORDER, start=65):
        path = resolve_path(default_index_path(year, index_name))
        exists = path.exists()
        row: dict[str, Any] = {
            "year": year,
            "record_type": "aligned_index",
            "feature_position_1based": idx_pos,
            "feature_name": index_name,
            "path": str(path),
            "exists": exists,
            "band_count": "",
            "band_count_ok": False,
            "crs": "",
            "crs_match_ae": False,
            "transform": "",
            "transform_match_ae": False,
            "width": "",
            "width_match_ae": False,
            "height": "",
            "height_match_ae": False,
            "nodata": "",
            "nodata_ok": False,
            "overall_ok": False,
            "notes": "",
        }

        if not exists:
            row["notes"] = "Aligned index raster missing"
            rows.append(row)
            continue

        with rasterio.open(path) as src:
            row["band_count"] = src.count
            row["band_count_ok"] = src.count == INDEX_EXPECTED_BANDS
            row["crs"] = str(src.crs)
            row["transform"] = str(src.transform)
            row["width"] = src.width
            row["height"] = src.height
            row["nodata"] = src.nodata
            row["nodata_ok"] = nodata_matches(src.nodata, INDEX_EXPECTED_NODATA)

            if ae_meta:
                row["crs_match_ae"] = str(src.crs) == ae_meta["crs"]
                row["transform_match_ae"] = affine_matches(src.transform, ae_meta["transform"])
                row["width_match_ae"] = src.width == ae_meta["width"]
                row["height_match_ae"] = src.height == ae_meta["height"]
            else:
                row["notes"] = "AE metadata unavailable"

            row["overall_ok"] = all(
                [
                    row["band_count_ok"],
                    row["crs_match_ae"],
                    row["transform_match_ae"],
                    row["width_match_ae"],
                    row["height_match_ae"],
                    row["nodata_ok"],
                ]
            )
        rows.append(row)

    for pos, feat_name in enumerate(FEATURE_ORDER, start=1):
        rows.append(
            {
                "year": year,
                "record_type": "feature_order",
                "feature_position_1based": pos,
                "feature_name": feat_name,
                "path": "",
                "exists": "",
                "band_count": "",
                "band_count_ok": "",
                "crs": "",
                "crs_match_ae": "",
                "transform": "",
                "transform_match_ae": "",
                "width": "",
                "width_match_ae": "",
                "height": "",
                "height_match_ae": "",
                "nodata": "",
                "nodata_ok": "",
                "overall_ok": True,
                "notes": "Locked inference feature order",
            }
        )

    return rows


def print_year_report(year: int, rows: list[dict[str, Any]]) -> None:
    ae_rows = [r for r in rows if r["record_type"] == "ae_reference"]
    idx_rows = [r for r in rows if r["record_type"] == "aligned_index"]

    ae_ok = bool(ae_rows) and bool(ae_rows[0]["overall_ok"])
    idx_ok_count = sum(bool(r["overall_ok"]) for r in idx_rows)
    idx_total = len(idx_rows)

    log(f"Year {year} validation summary")
    log(f"  AE reference OK        : {ae_ok}")
    log(f"  aligned indices OK     : {idx_ok_count}/{idx_total}")

    if ae_rows:
        ae = ae_rows[0]
        log(f"  AE path                : {ae['path']}")
        log(f"  AE bands               : {ae['band_count']} (expected {AE_EXPECTED_BANDS})")
        log(f"  AE nodata              : {ae['nodata']} (expected {AE_EXPECTED_NODATA})")

    failed = [r for r in idx_rows if not r["overall_ok"]]
    if failed:
        log("  failed aligned indices :")
        for r in failed:
            log(
                "    "
                f"{r['feature_name']}: exists={r['exists']}, "
                f"bands_ok={r['band_count_ok']}, crs_ok={r['crs_match_ae']}, "
                f"transform_ok={r['transform_match_ae']}, width_ok={r['width_match_ae']}, "
                f"height_ok={r['height_match_ae']}, nodata_ok={r['nodata_ok']}"
            )
    else:
        log("  all aligned indices passed")

    log("  locked feature order   :")
    log("    " + ", ".join(FEATURE_ORDER))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "year",
        "record_type",
        "feature_position_1based",
        "feature_name",
        "path",
        "exists",
        "band_count",
        "band_count_ok",
        "crs",
        "crs_match_ae",
        "transform",
        "transform_match_ae",
        "width",
        "width_match_ae",
        "height",
        "height_match_ae",
        "nodata",
        "nodata_ok",
        "overall_ok",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_csv = resolve_path(args.output_csv)

    all_rows: list[dict[str, Any]] = []
    for year in args.years:
        rows = validate_year(year)
        all_rows.extend(rows)
        print_year_report(year, rows)

    write_csv(output_csv, all_rows)
    log(f"Wrote CSV report: {output_csv}")


if __name__ == "__main__":
    main()
