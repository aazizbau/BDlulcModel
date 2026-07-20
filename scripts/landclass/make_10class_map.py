#!/usr/bin/env python3
"""
Merge existing maj_class/sub_class labels in Upazila landuse GPKGs into a new
10-class schema (coastal climate/policy oriented), and write a new GPKG.

Usage:
  python scripts/landclass/make_10class_map.py --upazila manpura --output assets/maps/manpura_10class.gpkg

Notes:
- Expects fields named: maj_class, sub_class (case-insensitive fallbacks supported).
- Preserves all original attributes and geometries; adds:
    - class10_id (1..10)
    - class10_name
    - class10_source ("maj_class/sub_class")
- Drops features that cannot be mapped only if --drop-unmapped is passed.
  Otherwise, keeps them and labels class10_name="UNMAPPED".

Reproduction and AOI adaptation
-------------------------------
Workflow role: Build or harmonize the ten-class coastal LULC classification layer.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--upazila``, ``--output``, ``--layer``, ``--out-layer``, ``--drop-unmapped``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace source classes and AOI paths, and explicitly document any new mapping into the ten-class scheme.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd


UPAZILA_TO_GPKG = {
    "manpura": Path("assets/maps/manpura_landuse.gpkg"),
    "betagi": Path("assets/maps/betagi_landuse.gpkg"),
    "amtali": Path("assets/maps/amtali_landuse.gpkg"),
    "bamna": Path("assets/maps/bamna_landuse.gpkg"),
    "haimchar": Path("assets/maps/haimchar_landuse.gpkg"),
}

# ----------------------------
# 10-class schema (final)
# ----------------------------
CLASS10 = {
    1: "Urban / Institutional Built-up",
    2: "Rural Settlement (Homestead Vegetation)",
    3: "Transport & Coastal Embankments",
    4: "Cropland (All Crop Intensities)",
    5: "Tree-based Agroforestry & Orchard",
    6: "Aquaculture & Inland Ponds",
    7: "Canals & Drainage Network",
    8: "Rivers & Estuarine Channels",
    9: "Mangrove Forest",
    10: "Bare / Exposed Coastal Land",
}

# ----------------------------
# Merge rules from your 5-upazila summaries
# ----------------------------

# Maj-class-only rules (if sub_class missing/unknown)
MAJ_RULES: Dict[str, int] = {
    # Built-up / institutional / POIs
    "administrative area": 1,
    "commercial area": 1,
    "educational institutions": 1,
    "industrial area": 1,
    "religious places": 1,
    "recreational places": 1,
    "monument": 1,
    "terminal": 1,
    # Roads/infrastructure
    "road": 3,
    # Agriculture major class -> split by sub_class when possible
    "agricultural land": 4,
    # Forest -> split by sub_class when possible
    "forest": 9,
    # Misc -> split by sub_class when possible
    "miscellaneous landuse": 10,
    # Water -> split by sub_class when possible
    "water bodies": 6,
    # Settlement -> split by sub_class when possible
    "settlement": 2,
}

# (maj_class, sub_class) specific overrides
# Keys are normalized (lower, stripped)
SUB_RULES: Dict[Tuple[str, str], int] = {
    # Settlement
    ("settlement", "rural settlement with homestead vegetation"): 2,
    ("settlement", "urban built-up area"): 1,
    ("settlement", "area under development"): 1,
    ("settlement", "open space"): 1,

    # Roads / transport
    ("road", "rural road"): 3,
    ("road", "upazila road"): 3,
    ("road", "zila road"): 3,
    ("road", "regional road"): 3,
    ("road", "access road"): 3,
    ("road", "right of way"): 3,
    ("road", "embankment cum road"): 3,
    ("road", "bridge"): 3,

    # Agriculture -> cropland
    ("agricultural land", "single crop land"): 4,
    ("agricultural land", "double crop land"): 4,
    ("agricultural land", "triple crop land"): 4,
    ("agricultural land", "cultivable fallow land"): 4,

    # Tree-based agro systems
    ("agricultural land", "agroforestry"): 5,
    ("agricultural land", "orchard"): 5,
    ("agricultural land", "betel leaf trellis"): 5,

    # Aquaculture / ponds
    ("agricultural land", "rice-cum-fish culture"): 6,
    ("water bodies", "pond"): 6,
    ("water bodies", "aquaculture"): 6,

    # Drainage
    ("water bodies", "ditch"): 7,
    ("water bodies", "canal"): 7,

    # Rivers
    ("water bodies", "river"): 8,

    # Mangroves
    ("forest", "mangrove plantation"): 9,

    # Bare / exposed land
    ("miscellaneous landuse", "mudflat"): 10,
    ("miscellaneous landuse", "sandbar/beach"): 10,
    ("miscellaneous landuse", "non-cultivable fallow land"): 10,
    ("miscellaneous landuse", "grass land"): 10,

    # Built-up explicit subclasses
    ("commercial area", "commercial area"): 1,
    ("commercial area", "brick field"): 1,
    ("commercial area", "gas field/power station"): 1,
    ("commercial area", "extraction/dumping/mining site"): 1,
    ("commercial area", "filling station"): 1,
    ("educational institutions", "school/college/madrasha"): 1,
    ("educational institutions", "educational institutions"): 1,
    ("administrative area", "office compound"): 1,
    ("industrial area", "rice mill /chatal"): 1,
    ("religious places", "mosque"): 1,
    ("religious places", "temple"): 1,
    ("religious places", "church"): 1,
    ("religious places", "eidgah"): 1,
    ("religious places", "graveyard"): 1,
    ("religious places", "religious places"): 1,
    ("recreational places", "playground"): 1,
    ("monument", "shaheed minar"): 1,
    ("monument", "other monuments"): 1,
    ("terminal", "boat ghat"): 1,
    ("terminal", "ferry ghat"): 1,
    ("terminal", "launch terminal"): 1,
    ("terminal", "helipad"): 1,
}


def _norm(x: Optional[str]) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().lower().split())


def _find_col(gdf: gpd.GeoDataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in gdf.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def map_to_class10(maj: str, sub: str) -> Tuple[Optional[int], str]:
    """Return (class_id, reason_string)."""
    maj_n = _norm(maj)
    sub_n = _norm(sub)

    key = (maj_n, sub_n)
    if key in SUB_RULES:
        cid = SUB_RULES[key]
        return cid, f"{maj_n}/{sub_n}"

    if maj_n in MAJ_RULES:
        cid = MAJ_RULES[maj_n]
        return cid, f"{maj_n}/(maj-default)"

    return None, f"{maj_n}/{sub_n}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upazila", required=True, choices=sorted(UPAZILA_TO_GPKG.keys()))
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument(
        "--layer",
        default=None,
        help="Optional layer name to read from GPKG. If omitted, reads first layer.",
    )
    ap.add_argument(
        "--out-layer",
        default="landuse_10class",
        help="Layer name to write in output GPKG.",
    )
    ap.add_argument(
        "--drop-unmapped",
        action="store_true",
        help="Drop features that do not match any rule.",
    )
    args = ap.parse_args()

    in_path = UPAZILA_TO_GPKG[args.upazila]
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input GPKG not found: {in_path}")

    if args.layer:
        gdf = gpd.read_file(in_path, layer=args.layer)
    else:
        gdf = gpd.read_file(in_path)

    if gdf.empty:
        raise RuntimeError(
            f"No features found in {in_path} (layer={args.layer or 'auto'})."
        )

    maj_col = _find_col(
        gdf, ["maj_class", "majclass", "major_class", "majorclass", "maj"]
    )
    sub_col = _find_col(gdf, ["sub_class", "subclass", "sub_classname", "sub"])

    if maj_col is None:
        raise KeyError(
            f"Could not find maj_class column in {in_path}. Columns: {list(gdf.columns)}"
        )
    if sub_col is None:
        gdf["_sub_class_tmp"] = ""
        sub_col = "_sub_class_tmp"

    class_ids: List[Optional[int]] = []
    reasons: List[str] = []
    names: List[str] = []

    for maj, sub in zip(gdf[maj_col], gdf[sub_col]):
        cid, reason = map_to_class10(str(maj), str(sub))
        class_ids.append(cid)
        reasons.append(reason)
        names.append("UNMAPPED" if cid is None else CLASS10[cid])

    gdf["class10_id"] = class_ids
    gdf["class10_name"] = names
    gdf["class10_source"] = reasons

    if args.drop_unmapped:
        gdf = gdf[gdf["class10_id"].notna()].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    gdf.to_file(out_path, layer=args.out_layer, driver="GPKG")

    mapped = gdf["class10_id"].notna().sum()
    total = len(gdf)
    print(f"[OK] Wrote: {out_path} (layer='{args.out_layer}')")
    print(f"[OK] Features: {total} | Mapped: {mapped} | Unmapped: {total - mapped}")
    print("Class distribution:")
    dist = (
        gdf[gdf["class10_id"].notna()]
        .groupby(["class10_id", "class10_name"])
        .size()
        .sort_values(ascending=False)
    )
    for (cid, cname), n in dist.items():
        print(f"  {int(cid):>2} - {cname:<40} {n}")


if __name__ == "__main__":
    main()
