"""Shared plotting helpers for asymmetric confidence intervals.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Estimate confidence intervals by resampling the original spatial test blocks.

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
Regenerate block IDs, predictions, and selected-run metadata from the new AOI spatial split before resampling; never reuse this project's block inventory.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Import this helper from its parent workflow or an interactive check::

    import scripts.spatial_block_bootstrap.common.plot_utils
"""

from __future__ import annotations

import textwrap

import matplotlib.patheffects as pe
import numpy as np


TEXT_EFFECTS = [
    pe.Stroke(linewidth=2.5, foreground="white"),
    pe.Stroke(linewidth=0.5, foreground="0.15"),
    pe.Normal(),
]

MAIN_VALUE_COLOR = "#1A1A1A"
BOUND_VALUE_COLOR = "#8B3E2F"
MAIN_VALUE_EFFECTS = [
    pe.Stroke(linewidth=1.8, foreground="white"),
    pe.Normal(),
]
BOUND_VALUE_EFFECTS = [
    pe.Stroke(linewidth=1.3, foreground="white"),
    pe.Normal(),
]


def asymmetric_yerr(observed: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.vstack([observed - lower, upper - observed])


def add_ci_labels(ax, bars, observed, lower, upper, fontsize: float = 6.5) -> None:
    bound_fontsize = fontsize * 0.8
    for bar, obs, lo, hi in zip(bars, observed, lower, upper):
        x = bar.get_x() + bar.get_width() / 2.0
        ax.text(
            x,
            hi + 0.8,
            f"{hi:.1f}%",
            color=BOUND_VALUE_COLOR,
            ha="center",
            va="bottom",
            fontsize=bound_fontsize,
            path_effects=BOUND_VALUE_EFFECTS,
        )
        ax.text(
            x,
            obs,
            f"{obs:.1f}%",
            color=MAIN_VALUE_COLOR,
            ha="center",
            va="center",
            fontsize=fontsize,
            path_effects=MAIN_VALUE_EFFECTS,
        )
        ax.text(
            x,
            max(0.2, lo - 0.8),
            f"{lo:.1f}%",
            color=BOUND_VALUE_COLOR,
            ha="center",
            va="top",
            fontsize=bound_fontsize,
            path_effects=BOUND_VALUE_EFFECTS,
        )


def wrap_label(value: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width))
