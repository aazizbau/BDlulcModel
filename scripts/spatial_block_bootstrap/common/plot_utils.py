"""Shared plotting helpers for asymmetric confidence intervals."""

from __future__ import annotations

import textwrap

import matplotlib.patheffects as pe
import numpy as np


TEXT_EFFECTS = [
    pe.Stroke(linewidth=2.5, foreground="white"),
    pe.Stroke(linewidth=0.5, foreground="0.15"),
    pe.Normal(),
]


def asymmetric_yerr(observed: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.vstack([observed - lower, upper - observed])


def add_ci_labels(ax, bars, observed, lower, upper, fontsize: float = 6.5) -> None:
    for bar, obs, lo, hi in zip(bars, observed, lower, upper):
        x = bar.get_x() + bar.get_width() / 2.0
        ax.text(x, hi + 0.8, f"{hi:.1f}%", ha="center", va="bottom", fontsize=fontsize, path_effects=TEXT_EFFECTS)
        ax.text(x, obs, f"{obs:.1f}%", ha="center", va="center", fontsize=fontsize, path_effects=TEXT_EFFECTS)
        ax.text(x, max(0.2, lo - 0.8), f"{lo:.1f}%", ha="center", va="top", fontsize=fontsize, path_effects=TEXT_EFFECTS)


def wrap_label(value: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width))
