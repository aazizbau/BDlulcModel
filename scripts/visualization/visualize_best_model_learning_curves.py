#!/usr/bin/env python3
"""
Visualize epoch-by-epoch learning curves for the best MLP model.

The script plots:
- training and validation loss
- training accuracy, validation accuracy, and validation balanced accuracy

It first reads the run-local history file from:
    runs/<model-name>/history.csv

If that file is not available, it falls back to:
    outputs/master_training_with_outputs/all_history_rows.csv

Example
-------
python scripts/visualization/visualize_best_model_learning_curves.py \
    --model-name mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3 \
    --add-title \
    --output outputs/figures/best_model_learning_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3"
DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_MASTER_HISTORY = Path("outputs/master_training_with_outputs/all_history_rows.csv")
DEFAULT_PALETTE = Path("assets/color_palette_coastal_lulc.json")
DEFAULT_OUTPUT = Path("outputs/figures/best_model_learning_curves.png")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot best-model learning curves from training history.")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Run directory name / model identifier.")
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Directory containing run folders.")
    p.add_argument("--master-history", type=Path, default=DEFAULT_MASTER_HISTORY, help="Fallback long history CSV.")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE, help="Project color palette JSON.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    p.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    p.add_argument("--add-title", action="store_true", help="Add model name as a figure title.")
    return p.parse_args()


def load_palette(path: Path) -> dict[str, str]:
    if not path.exists():
        return {
            "sand": "#FBEDD4",
            "deep_slate": "#314245",
            "teal_blue": "#3D848F",
            "coral": "#FF8973",
            "olive": "#808F54",
            "dust_rose": "#DBA796",
            "mist_gray": "#B6B5B8",
        }
    data = json.loads(path.read_text())
    return data.get("colors", {})


def load_history(model_name: str, runs_root: Path, master_history: Path) -> tuple[pd.DataFrame, Path]:
    run_history = runs_root / model_name / "history.csv"
    if run_history.exists():
        df = pd.read_csv(run_history)
        return df, run_history

    if not master_history.exists():
        raise FileNotFoundError(
            f"History not found at {run_history} and fallback CSV not found at {master_history}"
        )

    df = pd.read_csv(master_history)
    if "run_dir" not in df.columns:
        raise ValueError(f"Fallback history CSV lacks required column 'run_dir': {master_history}")

    mask = df["run_dir"].astype(str).str.endswith(model_name)
    df = df[mask].copy()
    if df.empty:
        raise ValueError(f"No rows for model '{model_name}' in {master_history}")

    return df, master_history


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def plot_learning_curves(
    df: pd.DataFrame,
    model_name: str,
    source: Path,
    output: Path,
    palette: dict[str, str],
    dpi: int,
    add_title: bool,
) -> None:
    require_columns(df, ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"], source)

    df = df.copy()
    df["epoch"] = numeric_series(df, "epoch")
    df = df.sort_values("epoch")
    df = df[np.isfinite(df["epoch"])]
    if df.empty:
        raise ValueError(f"No valid epoch rows found in {source}")

    fig_bg = palette.get("sand", "#FBEDD4")
    axis_color = palette.get("deep_slate", "#314245")
    train_color = palette.get("teal_blue", "#3D848F")
    val_color = palette.get("coral", "#FF8973")
    balanced_color = palette.get("olive", "#808F54")
    grid_color = palette.get("mist_gray", "#B6B5B8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi, facecolor=fig_bg)
    for ax in axes:
        ax.set_facecolor(fig_bg)
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.55)
        ax.tick_params(colors=axis_color, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(axis_color)
            spine.set_linewidth(0.9)

    epoch = numeric_series(df, "epoch")

    axes[0].plot(epoch, numeric_series(df, "train_loss"), color=train_color, linewidth=2.2, label="Training loss")
    axes[0].plot(epoch, numeric_series(df, "val_loss"), color=val_color, linewidth=2.2, label="Validation loss")
    axes[0].set_xlabel("Epoch", color=axis_color, fontsize=11)
    axes[0].set_ylabel("Loss", color=axis_color, fontsize=11)
    axes[0].set_title("Image (A) Loss", color=axis_color, fontsize=13, fontweight="bold", pad=10)
    axes[0].legend(frameon=False, fontsize=10)

    axes[1].plot(epoch, numeric_series(df, "train_acc"), color=train_color, linewidth=2.2, label="Training accuracy")
    axes[1].plot(epoch, numeric_series(df, "val_acc"), color=val_color, linewidth=2.2, label="Validation accuracy")
    if "val_balanced_acc" in df.columns and numeric_series(df, "val_balanced_acc").notna().any():
        axes[1].plot(
            epoch,
            numeric_series(df, "val_balanced_acc"),
            color=balanced_color,
            linewidth=2.0,
            label="Validation balanced accuracy",
        )
    axes[1].set_xlabel("Epoch", color=axis_color, fontsize=11)
    axes[1].set_ylabel("Accuracy", color=axis_color, fontsize=11)
    axes[1].set_title("Image (B) Accuracy", color=axis_color, fontsize=13, fontweight="bold", pad=10)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=10)

    if add_title:
        fig.suptitle(model_name, color=axis_color, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = resolve_path(args.runs_root)
    master_history = resolve_path(args.master_history)
    palette_path = resolve_path(args.palette)
    output = resolve_path(args.output)

    df, source = load_history(args.model_name, runs_root, master_history)
    palette = load_palette(palette_path)

    plot_learning_curves(
        df=df,
        model_name=args.model_name,
        source=source,
        output=output,
        palette=palette,
        dpi=args.dpi,
        add_title=args.add_title,
    )

    print(f"History source: {source}")
    print(f"Epochs plotted : {len(df)}")
    print(f"Saved figure   : {output}")


if __name__ == "__main__":
    main()
