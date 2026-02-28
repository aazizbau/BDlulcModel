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
