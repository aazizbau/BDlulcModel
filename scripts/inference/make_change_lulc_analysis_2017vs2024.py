#!/usr/bin/env python3
"""
Create blockwise LULC change analysis products from 2017 and 2024 inference outputs.

Inputs
------
- outputs/inference/2017/lulc_class_2017.tif
- outputs/inference/2024/lulc_class_2024.tif
- outputs/inference/2017/uncertainty_2017.tif
- outputs/inference/2024/uncertainty_2024.tif
- assets/maps/bd_coastal_zones.gpkg

Outputs
-------
- outputs/inference/change_analysis/change_binary_2017_vs_2024.tif
- outputs/inference/change_analysis/transition_code_2017_to_2024.tif
- outputs/inference/change_analysis/uncertainty_mean_2017_2024.tif
- outputs/inference/change_analysis/uncertainty_max_2017_2024.tif
- outputs/inference/change_analysis/coastal_zone_id.tif
- outputs/inference/change_analysis/analysis_summary.csv
- outputs/inference/change_analysis/change_summary_overall.csv
- outputs/inference/change_analysis/change_summary_by_zone.csv
- outputs/inference/change_analysis/class_area_summary_overall.csv
- outputs/inference/change_analysis/class_area_summary_by_zone.csv
- outputs/inference/change_analysis/transition_matrix_overall_long.csv
- outputs/inference/change_analysis/transition_matrix_by_zone_long.csv
- outputs/inference/change_analysis/uncertainty_summary_overall.csv
- outputs/inference/change_analysis/uncertainty_summary_by_zone.csv
- outputs/inference/change_analysis/zone_lookup.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR_DEFAULT = Path("outputs/inference/change_analysis")
ZONE_MAP_DEFAULT = Path("assets/maps/bd_coastal_zones.gpkg")

VALID_CLASSES = list(range(1, 11))
CLASS_NODATA = 0
CHANGE_NODATA = 255
TRANSITION_NODATA = 0
ZONE_NODATA = 0
UNC_NODATA = -9999.0

ZONE_NAME_MAP = {
    "western": "Western Zone",
    "central": "Central Zone",
    "eastern": "Eastern Zone",
}


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make overall and zone-wise LULC change analysis products for 2017 vs 2024."
    )
    p.add_argument("--class-2017", type=Path, default=Path("outputs/inference/2017/lulc_class_2017.tif"))
    p.add_argument("--class-2024", type=Path, default=Path("outputs/inference/2024/lulc_class_2024.tif"))
    p.add_argument("--uncertainty-2017", type=Path, default=Path("outputs/inference/2017/uncertainty_2017.tif"))
    p.add_argument("--uncertainty-2024", type=Path, default=Path("outputs/inference/2024/uncertainty_2024.tif"))
    p.add_argument("--zone-map", type=Path, default=ZONE_MAP_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=OUTDIR_DEFAULT)
    return p.parse_args()


def choose_zone_field(gdf: gpd.GeoDataFrame) -> str:
    for col in ["zone", "Zone", "ZONE", "zone_name", "ZONE_NAME", "name", "Name", "NAME"]:
        if col in gdf.columns:
            return col
    raise SystemExit(f"Could not determine zone label field from columns: {list(gdf.columns)}")


def same_grid(a: rasterio.DatasetReader, b: rasterio.DatasetReader, tol: float = 1e-9) -> bool:
    if a.crs != b.crs:
        return False
    if a.width != b.width or a.height != b.height:
        return False
    ta, tb = a.transform, b.transform
    return (
        abs(ta.a - tb.a) <= tol
        and abs(ta.b - tb.b) <= tol
        and abs(ta.c - tb.c) <= tol
        and abs(ta.d - tb.d) <= tol
        and abs(ta.e - tb.e) <= tol
        and abs(ta.f - tb.f) <= tol
    )


def write_csv(path: Path, rows: List[dict], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def zone_lookup_rows(zone_lookup: Dict[int, str]) -> List[dict]:
    return [{"zone_id": zid, "zone_name": zname} for zid, zname in sorted(zone_lookup.items())]


def init_counts(zone_ids: Iterable[int]) -> tuple[dict, dict, dict, dict]:
    overall_class = {2017: {c: 0 for c in VALID_CLASSES}, 2024: {c: 0 for c in VALID_CLASSES}}
    zone_class = {z: {2017: {c: 0 for c in VALID_CLASSES}, 2024: {c: 0 for c in VALID_CLASSES}} for z in zone_ids}

    overall_transition = {(c17, c24): 0 for c17 in VALID_CLASSES for c24 in VALID_CLASSES}
    zone_transition = {z: {(c17, c24): 0 for c17 in VALID_CLASSES for c24 in VALID_CLASSES} for z in zone_ids}
    return overall_class, zone_class, overall_transition, zone_transition


def init_change_stats(zone_ids: Iterable[int]) -> tuple[dict, dict]:
    overall = {"valid_pixels": 0, "changed_pixels": 0, "unchanged_pixels": 0}
    by_zone = {z: {"valid_pixels": 0, "changed_pixels": 0, "unchanged_pixels": 0} for z in zone_ids}
    return overall, by_zone


def init_uncertainty_stats(zone_ids: Iterable[int]) -> tuple[dict, dict]:
    def empty() -> dict:
        return {
            "2017_count": 0,
            "2017_sum": 0.0,
            "2024_count": 0,
            "2024_sum": 0.0,
            "mean_count": 0,
            "mean_sum": 0.0,
            "max_count": 0,
            "max_sum": 0.0,
            "changed_count": 0,
            "changed_sum": 0.0,
            "unchanged_count": 0,
            "unchanged_sum": 0.0,
        }

    overall = empty()
    by_zone = {z: empty() for z in zone_ids}
    return overall, by_zone


def rasterize_zones_window(
    zone_gdf: gpd.GeoDataFrame,
    zone_id_lookup: Dict[str, int],
    window,
    transform,
    out_shape: tuple[int, int],
) -> np.ndarray:
    shapes = []
    for _, row in zone_gdf.iterrows():
        zone_key = str(row["_zone_key"])
        shapes.append((row.geometry, zone_id_lookup[zone_key]))
    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=ZONE_NODATA,
        dtype="uint8",
        all_touched=False,
    )


def main() -> None:
    args = parse_args()
    class_2017_path = resolve_path(args.class_2017)
    class_2024_path = resolve_path(args.class_2024)
    uncertainty_2017_path = resolve_path(args.uncertainty_2017)
    uncertainty_2024_path = resolve_path(args.uncertainty_2024)
    zone_map_path = resolve_path(args.zone_map)
    outdir = resolve_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for path in [class_2017_path, class_2024_path, uncertainty_2017_path, uncertainty_2024_path, zone_map_path]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    zone_gdf = gpd.read_file(zone_map_path)
    if zone_gdf.empty:
        raise SystemExit(f"Zone map is empty: {zone_map_path}")

    zone_field = choose_zone_field(zone_gdf)
    zone_gdf["_zone_key"] = zone_gdf[zone_field].astype(str).str.strip().str.lower()
    zone_keys = sorted(zone_gdf["_zone_key"].unique().tolist())
    zone_id_lookup = {key: i + 1 for i, key in enumerate(zone_keys)}
    zone_name_lookup = {zone_id_lookup[k]: ZONE_NAME_MAP.get(k, k.title()) for k in zone_keys}

    with rasterio.open(class_2017_path) as cls17, \
        rasterio.open(class_2024_path) as cls24, \
        rasterio.open(uncertainty_2017_path) as unc17, \
        rasterio.open(uncertainty_2024_path) as unc24:

        if cls17.crs is None:
            raise SystemExit("Class 2017 raster has no CRS.")
        if not same_grid(cls17, cls24):
            raise SystemExit("2017 and 2024 class rasters do not share the same grid.")
        if not same_grid(cls17, unc17):
            raise SystemExit("2017 class and uncertainty rasters do not share the same grid.")
        if not same_grid(cls24, unc24):
            raise SystemExit("2024 class and uncertainty rasters do not share the same grid.")

        if zone_gdf.crs != cls17.crs:
            zone_gdf = zone_gdf.to_crs(cls17.crs)

        pixel_area_m2 = abs(cls17.transform.a * cls17.transform.e)

        overall_class, zone_class, overall_transition, zone_transition = init_counts(zone_name_lookup.keys())
        overall_change, zone_change = init_change_stats(zone_name_lookup.keys())
        overall_unc, zone_unc = init_uncertainty_stats(zone_name_lookup.keys())

        change_binary_path = outdir / "change_binary_2017_vs_2024.tif"
        transition_code_path = outdir / "transition_code_2017_to_2024.tif"
        uncertainty_mean_path = outdir / "uncertainty_mean_2017_2024.tif"
        uncertainty_max_path = outdir / "uncertainty_max_2017_2024.tif"
        zone_id_raster_path = outdir / "coastal_zone_id.tif"

        class_profile = cls17.profile.copy()
        class_profile.update({"count": 1, "dtype": "uint8", "nodata": CHANGE_NODATA, "compress": "ZSTD", "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"})

        transition_profile = cls17.profile.copy()
        transition_profile.update({"count": 1, "dtype": "uint16", "nodata": TRANSITION_NODATA, "compress": "ZSTD", "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"})

        float_profile = unc17.profile.copy()
        float_profile.update({"count": 1, "dtype": "float32", "nodata": UNC_NODATA, "compress": "ZSTD", "predictor": 3, "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"})

        zone_profile = cls17.profile.copy()
        zone_profile.update({"count": 1, "dtype": "uint8", "nodata": ZONE_NODATA, "compress": "ZSTD", "tiled": True, "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"})

        total_windows = sum(1 for _ in cls17.block_windows(1))
        processed_windows = 0

        with rasterio.open(change_binary_path, "w", **class_profile) as change_dst, \
            rasterio.open(transition_code_path, "w", **transition_profile) as transition_dst, \
            rasterio.open(uncertainty_mean_path, "w", **float_profile) as unc_mean_dst, \
            rasterio.open(uncertainty_max_path, "w", **float_profile) as unc_max_dst, \
            rasterio.open(zone_id_raster_path, "w", **zone_profile) as zone_dst:

            for _, window in cls17.block_windows(1):
                processed_windows += 1
                arr17 = cls17.read(1, window=window)
                arr24 = cls24.read(1, window=window)
                u17 = unc17.read(1, window=window).astype(np.float32)
                u24 = unc24.read(1, window=window).astype(np.float32)

                h = int(window.height)
                w = int(window.width)
                win_transform = rasterio.windows.transform(window, cls17.transform)
                zone_arr = rasterize_zones_window(zone_gdf, zone_id_lookup, window, win_transform, (h, w))

                valid = np.isin(arr17, VALID_CLASSES) & np.isin(arr24, VALID_CLASSES)
                changed = valid & (arr17 != arr24)
                unchanged = valid & (arr17 == arr24)

                u17_valid = np.isfinite(u17) & (u17 != float(UNC_NODATA))
                u24_valid = np.isfinite(u24) & (u24 != float(UNC_NODATA))
                unc_valid = valid & u17_valid & u24_valid

                mean_unc = np.full((h, w), UNC_NODATA, dtype=np.float32)
                max_unc = np.full((h, w), UNC_NODATA, dtype=np.float32)
                mean_unc[unc_valid] = ((u17[unc_valid] + u24[unc_valid]) / 2.0).astype(np.float32)
                max_unc[unc_valid] = np.maximum(u17[unc_valid], u24[unc_valid]).astype(np.float32)

                change_out = np.full((h, w), CHANGE_NODATA, dtype=np.uint8)
                change_out[valid] = 0
                change_out[changed] = 1

                transition_out = np.full((h, w), TRANSITION_NODATA, dtype=np.uint16)
                transition_out[valid] = (arr17[valid].astype(np.uint16) * 100 + arr24[valid].astype(np.uint16))

                change_dst.write(change_out, 1, window=window)
                transition_dst.write(transition_out, 1, window=window)
                unc_mean_dst.write(mean_unc, 1, window=window)
                unc_max_dst.write(max_unc, 1, window=window)
                zone_dst.write(zone_arr.astype(np.uint8), 1, window=window)

                if np.any(valid):
                    vals17 = arr17[valid]
                    vals24 = arr24[valid]
                    for cls in VALID_CLASSES:
                        overall_class[2017][cls] += int(np.sum(vals17 == cls))
                        overall_class[2024][cls] += int(np.sum(vals24 == cls))

                    pairs = np.stack([vals17, vals24], axis=1)
                    for c17, c24 in pairs:
                        overall_transition[(int(c17), int(c24))] += 1

                    overall_change["valid_pixels"] += int(valid.sum())
                    overall_change["changed_pixels"] += int(changed.sum())
                    overall_change["unchanged_pixels"] += int(unchanged.sum())

                if np.any(unc_valid):
                    overall_unc["2017_count"] += int(np.sum(valid & u17_valid))
                    overall_unc["2017_sum"] += float(u17[valid & u17_valid].sum())
                    overall_unc["2024_count"] += int(np.sum(valid & u24_valid))
                    overall_unc["2024_sum"] += float(u24[valid & u24_valid].sum())
                    overall_unc["mean_count"] += int(np.sum(unc_valid))
                    overall_unc["mean_sum"] += float(mean_unc[unc_valid].sum())
                    overall_unc["max_count"] += int(np.sum(unc_valid))
                    overall_unc["max_sum"] += float(max_unc[unc_valid].sum())
                    overall_unc["changed_count"] += int(np.sum(changed & unc_valid))
                    overall_unc["changed_sum"] += float(mean_unc[changed & unc_valid].sum())
                    overall_unc["unchanged_count"] += int(np.sum(unchanged & unc_valid))
                    overall_unc["unchanged_sum"] += float(mean_unc[unchanged & unc_valid].sum())

                for zone_id in zone_name_lookup.keys():
                    zone_mask = zone_arr == zone_id
                    zone_valid = valid & zone_mask
                    zone_unc_valid = unc_valid & zone_mask
                    zone_changed = changed & zone_mask
                    zone_unchanged = unchanged & zone_mask

                    if np.any(zone_valid):
                        vals17 = arr17[zone_valid]
                        vals24 = arr24[zone_valid]
                        for cls in VALID_CLASSES:
                            zone_class[zone_id][2017][cls] += int(np.sum(vals17 == cls))
                            zone_class[zone_id][2024][cls] += int(np.sum(vals24 == cls))
                        pairs = np.stack([vals17, vals24], axis=1)
                        for c17, c24 in pairs:
                            zone_transition[zone_id][(int(c17), int(c24))] += 1
                        zone_change[zone_id]["valid_pixels"] += int(zone_valid.sum())
                        zone_change[zone_id]["changed_pixels"] += int(zone_changed.sum())
                        zone_change[zone_id]["unchanged_pixels"] += int(zone_unchanged.sum())

                    if np.any(zone_unc_valid):
                        zone_unc[zone_id]["2017_count"] += int(np.sum(zone_valid & u17_valid))
                        zone_unc[zone_id]["2017_sum"] += float(u17[zone_valid & u17_valid].sum())
                        zone_unc[zone_id]["2024_count"] += int(np.sum(zone_valid & u24_valid))
                        zone_unc[zone_id]["2024_sum"] += float(u24[zone_valid & u24_valid].sum())
                        zone_unc[zone_id]["mean_count"] += int(np.sum(zone_unc_valid))
                        zone_unc[zone_id]["mean_sum"] += float(mean_unc[zone_unc_valid].sum())
                        zone_unc[zone_id]["max_count"] += int(np.sum(zone_unc_valid))
                        zone_unc[zone_id]["max_sum"] += float(max_unc[zone_unc_valid].sum())
                        zone_unc[zone_id]["changed_count"] += int(np.sum(zone_changed & zone_unc_valid))
                        zone_unc[zone_id]["changed_sum"] += float(mean_unc[zone_changed & zone_unc_valid].sum())
                        zone_unc[zone_id]["unchanged_count"] += int(np.sum(zone_unchanged & zone_unc_valid))
                        zone_unc[zone_id]["unchanged_sum"] += float(mean_unc[zone_unchanged & zone_unc_valid].sum())

                if processed_windows % 250 == 0 or processed_windows == total_windows:
                    log(f"Processed windows: {processed_windows}/{total_windows}")

    zone_lookup_csv = outdir / "zone_lookup.csv"
    write_csv(zone_lookup_csv, zone_lookup_rows(zone_name_lookup))

    analysis_summary_csv = outdir / "analysis_summary.csv"
    change_summary_overall_csv = outdir / "change_summary_overall.csv"
    change_summary_by_zone_csv = outdir / "change_summary_by_zone.csv"
    class_area_overall_csv = outdir / "class_area_summary_overall.csv"
    class_area_by_zone_csv = outdir / "class_area_summary_by_zone.csv"
    transition_overall_csv = outdir / "transition_matrix_overall_long.csv"
    transition_by_zone_csv = outdir / "transition_matrix_by_zone_long.csv"
    uncertainty_overall_csv = outdir / "uncertainty_summary_overall.csv"
    uncertainty_by_zone_csv = outdir / "uncertainty_summary_by_zone.csv"

    analysis_summary_rows = [
        {"metric": "pixel_area_m2", "value": pixel_area_m2},
        {"metric": "valid_pixels_overall", "value": overall_change["valid_pixels"]},
        {"metric": "changed_pixels_overall", "value": overall_change["changed_pixels"]},
        {"metric": "unchanged_pixels_overall", "value": overall_change["unchanged_pixels"]},
        {"metric": "change_fraction_overall", "value": (overall_change["changed_pixels"] / overall_change["valid_pixels"]) if overall_change["valid_pixels"] > 0 else ""},
        {"metric": "zone_count", "value": len(zone_name_lookup)},
        {"metric": "change_binary_tif", "value": str(change_binary_path)},
        {"metric": "transition_code_tif", "value": str(transition_code_path)},
        {"metric": "uncertainty_mean_tif", "value": str(uncertainty_mean_path)},
        {"metric": "uncertainty_max_tif", "value": str(uncertainty_max_path)},
        {"metric": "coastal_zone_id_tif", "value": str(zone_id_raster_path)},
    ]
    write_csv(analysis_summary_csv, analysis_summary_rows, fieldnames=["metric", "value"])

    change_overall_rows = [
        {
            "scope": "overall",
            "valid_pixels": overall_change["valid_pixels"],
            "changed_pixels": overall_change["changed_pixels"],
            "unchanged_pixels": overall_change["unchanged_pixels"],
            "changed_area_m2": overall_change["changed_pixels"] * pixel_area_m2,
            "changed_area_km2": overall_change["changed_pixels"] * pixel_area_m2 / 1_000_000.0,
            "unchanged_area_m2": overall_change["unchanged_pixels"] * pixel_area_m2,
            "unchanged_area_km2": overall_change["unchanged_pixels"] * pixel_area_m2 / 1_000_000.0,
            "change_fraction": (overall_change["changed_pixels"] / overall_change["valid_pixels"]) if overall_change["valid_pixels"] > 0 else "",
        }
    ]
    write_csv(change_summary_overall_csv, change_overall_rows)

    change_zone_rows: List[dict] = []
    for zone_id, zone_name in zone_name_lookup.items():
        st = zone_change[zone_id]
        change_zone_rows.append(
            {
                "zone_id": zone_id,
                "zone_name": zone_name,
                "valid_pixels": st["valid_pixels"],
                "changed_pixels": st["changed_pixels"],
                "unchanged_pixels": st["unchanged_pixels"],
                "changed_area_m2": st["changed_pixels"] * pixel_area_m2,
                "changed_area_km2": st["changed_pixels"] * pixel_area_m2 / 1_000_000.0,
                "unchanged_area_m2": st["unchanged_pixels"] * pixel_area_m2,
                "unchanged_area_km2": st["unchanged_pixels"] * pixel_area_m2 / 1_000_000.0,
                "change_fraction": (st["changed_pixels"] / st["valid_pixels"]) if st["valid_pixels"] > 0 else "",
            }
        )
    write_csv(change_summary_by_zone_csv, change_zone_rows)

    class_area_overall_rows: List[dict] = []
    for year in [2017, 2024]:
        for cls in VALID_CLASSES:
            pixels = overall_class[year][cls]
            class_area_overall_rows.append(
                {
                    "scope": "overall",
                    "year": year,
                    "class_id": cls,
                    "pixel_count": pixels,
                    "area_m2": pixels * pixel_area_m2,
                    "area_km2": pixels * pixel_area_m2 / 1_000_000.0,
                }
            )
    write_csv(class_area_overall_csv, class_area_overall_rows)

    class_area_zone_rows: List[dict] = []
    for zone_id, zone_name in zone_name_lookup.items():
        for year in [2017, 2024]:
            for cls in VALID_CLASSES:
                pixels = zone_class[zone_id][year][cls]
                class_area_zone_rows.append(
                    {
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "year": year,
                        "class_id": cls,
                        "pixel_count": pixels,
                        "area_m2": pixels * pixel_area_m2,
                        "area_km2": pixels * pixel_area_m2 / 1_000_000.0,
                    }
                )
    write_csv(class_area_by_zone_csv, class_area_zone_rows)

    transition_overall_rows: List[dict] = []
    for c17 in VALID_CLASSES:
        for c24 in VALID_CLASSES:
            pixels = overall_transition[(c17, c24)]
            transition_overall_rows.append(
                {
                    "scope": "overall",
                    "class_2017": c17,
                    "class_2024": c24,
                    "pixel_count": pixels,
                    "area_m2": pixels * pixel_area_m2,
                    "area_km2": pixels * pixel_area_m2 / 1_000_000.0,
                    "changed_flag": int(c17 != c24),
                }
            )
    write_csv(transition_overall_csv, transition_overall_rows)

    transition_zone_rows: List[dict] = []
    for zone_id, zone_name in zone_name_lookup.items():
        for c17 in VALID_CLASSES:
            for c24 in VALID_CLASSES:
                pixels = zone_transition[zone_id][(c17, c24)]
                transition_zone_rows.append(
                    {
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "class_2017": c17,
                        "class_2024": c24,
                        "pixel_count": pixels,
                        "area_m2": pixels * pixel_area_m2,
                        "area_km2": pixels * pixel_area_m2 / 1_000_000.0,
                        "changed_flag": int(c17 != c24),
                    }
                )
    write_csv(transition_by_zone_csv, transition_zone_rows)

    def unc_row(scope: str, zone_id: int | str, zone_name: str, stats: dict) -> dict:
        return {
            "scope": scope,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "mean_uncertainty_2017": (stats["2017_sum"] / stats["2017_count"]) if stats["2017_count"] > 0 else "",
            "mean_uncertainty_2024": (stats["2024_sum"] / stats["2024_count"]) if stats["2024_count"] > 0 else "",
            "mean_uncertainty_2017_2024": (stats["mean_sum"] / stats["mean_count"]) if stats["mean_count"] > 0 else "",
            "max_uncertainty_mean": (stats["max_sum"] / stats["max_count"]) if stats["max_count"] > 0 else "",
            "mean_uncertainty_changed_pixels": (stats["changed_sum"] / stats["changed_count"]) if stats["changed_count"] > 0 else "",
            "mean_uncertainty_unchanged_pixels": (stats["unchanged_sum"] / stats["unchanged_count"]) if stats["unchanged_count"] > 0 else "",
            "count_2017": stats["2017_count"],
            "count_2024": stats["2024_count"],
            "count_mean": stats["mean_count"],
            "count_changed": stats["changed_count"],
            "count_unchanged": stats["unchanged_count"],
        }

    uncertainty_overall_rows = [unc_row("overall", "", "Overall", overall_unc)]
    write_csv(uncertainty_overall_csv, uncertainty_overall_rows)

    uncertainty_zone_rows = [unc_row("zone", zone_id, zone_name_lookup[zone_id], zone_unc[zone_id]) for zone_id in zone_name_lookup.keys()]
    write_csv(uncertainty_by_zone_csv, uncertainty_zone_rows)

    log(f"Saved: {change_binary_path}")
    log(f"Saved: {transition_code_path}")
    log(f"Saved: {uncertainty_mean_path}")
    log(f"Saved: {uncertainty_max_path}")
    log(f"Saved: {zone_id_raster_path}")
    log(f"Saved: {analysis_summary_csv}")
    log(f"Saved: {change_summary_overall_csv}")
    log(f"Saved: {change_summary_by_zone_csv}")
    log(f"Saved: {class_area_overall_csv}")
    log(f"Saved: {class_area_by_zone_csv}")
    log(f"Saved: {transition_overall_csv}")
    log(f"Saved: {transition_by_zone_csv}")
    log(f"Saved: {uncertainty_overall_csv}")
    log(f"Saved: {uncertainty_by_zone_csv}")
    log(f"Saved: {zone_lookup_csv}")


if __name__ == "__main__":
    main()
