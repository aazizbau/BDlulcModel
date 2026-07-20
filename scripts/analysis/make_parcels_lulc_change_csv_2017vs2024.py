#!/usr/bin/env python3
"""
Create a parcel-level 2017 vs 2024 LULC change CSV for a target upazila.

Inputs
------
- assets/maps/<upazila>_parcels_lulc_2017.gpkg
- assets/maps/<upazila>_parcels_lulc_2024.gpkg

Output
------
- outputs/inference/change_analysis/<upazila>_parcels_lulc_change_2017vs2024.csv

Example
-------
python scripts/analysis/make_parcels_lulc_change_csv_2017vs2024.py \
    --upazila-gpkg bamna

Reproduction and AOI adaptation
-------------------------------
Workflow role: Derive quantitative summaries, accuracy assessments, or change statistics from prepared model outputs.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--upazila-gpkg``, ``--input-2017``, ``--input-2024``, ``--output``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace input result paths with outputs generated for the new AOI, and retain the same class-ID definitions when comparing results.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPS_ROOT = Path("assets/maps")
DEFAULT_OUTPUT_ROOT = Path("outputs/inference/change_analysis")
UPAZILA_CHOICES = ("bamna", "betagi", "manpura", "amtali")
YEAR_2017 = 2017
YEAR_2024 = 2024
AREA_CRS_EPSG = 32646
ACRE_PER_M2 = 0.00024710538146716534

PREFERRED_KEY_COLUMNS = ("parcel_id", "OBJECTID", "fid", "id")
VARYING_LULC_FIELDS = ("lulc_class", "lulc_name", "major_px", "valid_px", "major_frac")
GEOMETRY_DERIVED_KEY = "__geom_key__"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_input_path(upazila: str, year: int) -> Path:
    return DEFAULT_MAPS_ROOT / f"{upazila}_parcels_lulc_{year}.gpkg"


def default_output_path(upazila: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{upazila}_parcels_lulc_change_2017vs2024.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export parcel LULC change attributes from 2017 and 2024 GPKGs to CSV.")
    parser.add_argument(
        "--upazila-gpkg",
        required=True,
        choices=UPAZILA_CHOICES,
        help="Upazila parcel LULC GPKG basename to use.",
    )
    parser.add_argument(
        "--input-2017",
        type=Path,
        default=None,
        help="Optional 2017 input GPKG override.",
    )
    parser.add_argument(
        "--input-2024",
        type=Path,
        default=None,
        help="Optional 2024 input GPKG override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV override.",
    )
    return parser.parse_args()


def geometry_hash(geom) -> str:
    return hashlib.sha1(geom.wkb).hexdigest()


def choose_merge_keys(gdf_2017: gpd.GeoDataFrame, gdf_2024: gpd.GeoDataFrame) -> tuple[list[str], str]:
    for col in PREFERRED_KEY_COLUMNS:
        if col in gdf_2017.columns and col in gdf_2024.columns:
            if (
                gdf_2017[col].notna().all()
                and gdf_2024[col].notna().all()
                and gdf_2017[col].is_unique
                and gdf_2024[col].is_unique
            ):
                return [col], f"preferred unique key: {col}"

    stable_common = sorted(
        (set(gdf_2017.columns) & set(gdf_2024.columns))
        - set(VARYING_LULC_FIELDS)
        - {gdf_2017.geometry.name, gdf_2024.geometry.name}
    )
    stable_common = [c for c in stable_common if not c.startswith("Shape_")]
    if stable_common:
        if not gdf_2017[stable_common].isna().any().any() and not gdf_2024[stable_common].isna().any().any():
            if not gdf_2017.duplicated(subset=stable_common).any() and not gdf_2024.duplicated(subset=stable_common).any():
                return stable_common, f"composite stable keys: {', '.join(stable_common)}"

    return [GEOMETRY_DERIVED_KEY], "geometry-derived hash key"


def add_area_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Parcel GPKG has no CRS.")
    work = gdf
    if work.crs.to_epsg() != AREA_CRS_EPSG:
        work = work.to_crs(epsg=AREA_CRS_EPSG)
    area_m2 = work.geometry.area.astype(float)
    out = gdf.copy()
    out["parcel_area_m2"] = area_m2.values
    out["parcel_area_ha"] = area_m2.values / 10_000.0
    out["parcel_area_acre"] = area_m2.values * ACRE_PER_M2
    return out


def prepare_dataframe(gdf: gpd.GeoDataFrame, year: int, merge_keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name, errors="ignore")).copy()
    rename_map = {
        "lulc_class": f"lulc_class_{year}",
        "lulc_name": f"lulc_name_{year}",
        "major_px": f"major_px_{year}",
        "valid_px": f"valid_px_{year}",
        "major_frac": f"major_frac_{year}",
    }
    return df.rename(columns=rename_map)


def collapse_common_fields(merged: pd.DataFrame, merge_keys: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    stable_unsuffixed: list[str] = []
    differing_common: list[str] = []

    suffix_2017 = f"_{YEAR_2017}"
    suffix_2024 = f"_{YEAR_2024}"
    common_bases = sorted(
        {
            c[: -len(suffix_2017)]
            for c in merged.columns
            if c.endswith(suffix_2017) and c[: -len(suffix_2017)] + suffix_2024 in merged.columns
        }
    )

    for base in common_bases:
        col_2017 = f"{base}_{YEAR_2017}"
        col_2024 = f"{base}_{YEAR_2024}"
        if base in VARYING_LULC_FIELDS:
            differing_common.append(base)
            continue
        same = merged[col_2017].equals(merged[col_2024])
        if same:
            merged[base] = merged[col_2017]
            merged.drop(columns=[col_2017, col_2024], inplace=True)
            if base not in merge_keys:
                stable_unsuffixed.append(base)
        else:
            differing_common.append(base)

    return merged, stable_unsuffixed, differing_common


def add_derived_change_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["changed_2017_2024"] = (df["lulc_class_2017"] != df["lulc_class_2024"]).astype(int)
    df["transition_code_2017_2024"] = df["lulc_class_2017"].astype("Int64").astype(str) + "_" + df["lulc_class_2024"].astype("Int64").astype(str)
    df["transition_label_2017_2024"] = df["lulc_name_2017"].astype(str) + " -> " + df["lulc_name_2024"].astype(str)
    df["major_frac_diff_2024_minus_2017"] = df["major_frac_2024"] - df["major_frac_2017"]
    df["major_px_diff_2024_minus_2017"] = df["major_px_2024"] - df["major_px_2017"]
    df["valid_px_diff_2024_minus_2017"] = df["valid_px_2024"] - df["valid_px_2017"]
    return df


def ordered_columns(df: pd.DataFrame, merge_keys: list[str], stable_unsuffixed: list[str], differing_common: list[str]) -> list[str]:
    preferred_stable = []
    for col in merge_keys:
        if col in df.columns and col != GEOMETRY_DERIVED_KEY:
            preferred_stable.append(col)
    for col in stable_unsuffixed:
        if col in df.columns and col not in preferred_stable:
            preferred_stable.append(col)

    area_cols = [c for c in ("parcel_area_m2", "parcel_area_ha", "parcel_area_acre") if c in df.columns]
    lulc_cols = [
        "lulc_class_2017",
        "lulc_name_2017",
        "major_px_2017",
        "valid_px_2017",
        "major_frac_2017",
        "lulc_class_2024",
        "lulc_name_2024",
        "major_px_2024",
        "valid_px_2024",
        "major_frac_2024",
    ]
    derived_cols = [
        "changed_2017_2024",
        "transition_code_2017_2024",
        "transition_label_2017_2024",
        "major_frac_diff_2024_minus_2017",
        "major_px_diff_2024_minus_2017",
        "valid_px_diff_2024_minus_2017",
    ]

    differing_cols = []
    for base in differing_common:
        for year in (YEAR_2017, YEAR_2024):
            col = f"{base}_{year}"
            if col in df.columns:
                differing_cols.append(col)

    ordered = []
    for col in preferred_stable + area_cols + lulc_cols + derived_cols + differing_cols:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    for col in df.columns:
        if col not in ordered and col != GEOMETRY_DERIVED_KEY:
            ordered.append(col)
    return ordered


def main() -> None:
    args = parse_args()
    input_2017 = resolve_path(args.input_2017 or default_input_path(args.upazila_gpkg, YEAR_2017))
    input_2024 = resolve_path(args.input_2024 or default_input_path(args.upazila_gpkg, YEAR_2024))
    output_path = resolve_path(args.output or default_output_path(args.upazila_gpkg))

    if not input_2017.exists():
        raise FileNotFoundError(f"2017 input GPKG not found: {input_2017}")
    if not input_2024.exists():
        raise FileNotFoundError(f"2024 input GPKG not found: {input_2024}")

    gdf_2017 = gpd.read_file(input_2017)
    gdf_2024 = gpd.read_file(input_2024)
    if gdf_2017.empty or gdf_2024.empty:
        raise ValueError("One or both input GPKGs contain no features.")

    gdf_2017 = add_area_fields(gdf_2017)
    gdf_2024 = add_area_fields(gdf_2024)

    merge_keys, merge_strategy = choose_merge_keys(gdf_2017, gdf_2024)
    if GEOMETRY_DERIVED_KEY in merge_keys:
        gdf_2017[GEOMETRY_DERIVED_KEY] = gdf_2017.geometry.apply(geometry_hash)
        gdf_2024[GEOMETRY_DERIVED_KEY] = gdf_2024.geometry.apply(geometry_hash)

    df_2017 = prepare_dataframe(gdf_2017, YEAR_2017, merge_keys)
    df_2024 = prepare_dataframe(gdf_2024, YEAR_2024, merge_keys)

    merged = df_2017.merge(
        df_2024,
        on=merge_keys,
        how="inner",
        suffixes=(f"_{YEAR_2017}", f"_{YEAR_2024}"),
        validate="one_to_one",
    )

    merged, stable_unsuffixed, differing_common = collapse_common_fields(merged, merge_keys)
    merged = add_derived_change_fields(merged)
    cols = ordered_columns(merged, merge_keys, stable_unsuffixed, differing_common)
    final_df = merged[cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"Input 2017    : {input_2017}")
    print(f"Input 2024    : {input_2024}")
    print(f"Output CSV    : {output_path}")
    print(f"Merge strategy: {merge_strategy}")
    print(f"Rows          : {len(final_df):,}")
    print(f"Columns       : {len(final_df.columns)}")


if __name__ == "__main__":
    main()
