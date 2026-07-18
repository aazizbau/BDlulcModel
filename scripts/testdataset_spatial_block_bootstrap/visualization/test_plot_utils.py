"""Plot-label styling specific to the test-dataset bootstrap figures."""

from __future__ import annotations

import matplotlib.patheffects as pe

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
