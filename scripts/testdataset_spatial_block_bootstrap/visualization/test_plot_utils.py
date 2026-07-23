"""Plot-label styling specific to the test-dataset bootstrap figures.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Produce the test-selected spatial-block uncertainty analysis used for descriptive thesis results.

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
Regenerate test-selected run metadata and all block-level predictions for the new AOI. Treat test-selected intervals as descriptive, not unbiased model-selection evidence.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Import this helper from its parent workflow or an interactive check::

    import scripts.testdataset_spatial_block_bootstrap.visualization.test_plot_utils
"""

from __future__ import annotations

import matplotlib.patheffects as pe
from matplotlib.text import Text

from common.plot_utils import (
    BOUND_VALUE_COLOR,
    BOUND_VALUE_EFFECTS,
    MAIN_VALUE_COLOR,
    MAIN_VALUE_EFFECTS,
)


UPPER_BOUND_VALUE_EFFECTS = [
    pe.Stroke(linewidth=1.3, foreground="#FFF2A8"),
    pe.Normal(),
]


def double_height_figsize(figsize: tuple[float, float]) -> tuple[float, float]:
    """Return a figure size with the original width and twice the height."""
    return figsize[0], figsize[1] * 2.0


def double_figure_text(fig) -> None:
    """Double every existing text object's font size in a figure."""
    for text in fig.findobj(match=Text):
        text.set_fontsize(text.get_fontsize() * 2.0)


def add_ci_labels(ax, bars, observed, lower, upper, fontsize: float = 6.5) -> None:
    """Label observed values and CI bounds with a colored upper-bound halo."""
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
            path_effects=UPPER_BOUND_VALUE_EFFECTS,
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
