"""Reproduction and AOI adaptation
-------------------------------
Workflow role: Provide importable Earth Engine and AOI helper functions used by acquisition scripts.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is an imported helper module rather than a standalone command. Its public functions are exercised by the parent workflow scripts.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pass a replacement AOI geometry and Earth Engine project from the calling workflow; keep authentication and export settings outside reusable helpers.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Import this helper from its parent workflow or an interactive check::

    import src.gee.aoi
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import ee
import yaml


@dataclass(frozen=True)
class AOI:
    name: str
    coordinates: Dict[str, Dict[str, float]]

    def bbox_polygon(self) -> List[List[float]]:
        ul = self.coordinates["upper_left"]
        ur = self.coordinates["upper_right"]
        lr = self.coordinates["lower_right"]
        ll = self.coordinates["lower_left"]

        return [
            [float(ul["lon"]), float(ul["lat"])],
            [float(ll["lon"]), float(ll["lat"])],
            [float(lr["lon"]), float(lr["lat"])],
            [float(ur["lon"]), float(ur["lat"])],
            [float(ul["lon"]), float(ul["lat"])],
        ]

    def to_ee_geometry(self) -> ee.Geometry:
        return ee.Geometry.Polygon([self.bbox_polygon()])


def _validate_bbox(payload: Dict[str, Any], path: Path) -> None:
    if payload.get("selection") != "bbox":
        raise ValueError(f"Only selection=bbox is supported in {path}")

    coords = payload.get("coordinates")
    if not isinstance(coords, dict):
        raise ValueError(f"Missing/invalid coordinates in {path}")

    required = ("upper_left", "upper_right", "lower_right", "lower_left")
    for key in required:
        if key not in coords:
            raise ValueError(f"Missing coordinates.{key} in {path}")
        pt = coords[key]
        if not isinstance(pt, dict) or "lat" not in pt or "lon" not in pt:
            raise ValueError(f"coordinates.{key} must contain lat/lon in {path}")


def load_aoi(path: Path) -> AOI:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AOI config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid AOI YAML structure: {path}")

    _validate_bbox(payload, path)

    name = str(payload.get("name", "AOI_BBOX"))
    coordinates = payload["coordinates"]
    return AOI(name=name, coordinates=coordinates)
