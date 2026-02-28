from __future__ import annotations

import ee


def initialize_earth_engine(project: str | None = None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except ee.EEException:
        print("Authenticating to Google Earth Engine ...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    else:
        print("Google Earth Engine initialized.")
