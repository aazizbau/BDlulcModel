#!/usr/bin/env python3
"""
Merge final 10-seed experiment outputs into combined CSV files.

This script expects the family runners to have written seed folders under
outputs/final_10seed_experiment/<ModelFamily>/seed_###/.

Complete Example Run
--------------------
python scripts/final_10seed_experiment/merge_10seed_results.py \
    --output-root outputs/final_10seed_experiment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.experiment_constants import DEFAULT_OUTPUT_ROOT, MODEL_FAMILY_DISPLAY, MODEL_FAMILY_ORDER
from common.metric_utils import confusion_matrix_to_long, metrics_from_cm, read_confusion_matrix
from common.output_utils import read_json, write_csv, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge final 10-seed experiment outputs.")
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def collect_seed_run(seed_dir: Path) -> tuple[dict, pd.DataFrame] | None:
    summary_path = seed_dir / "summary.json"
    metadata_path = seed_dir / "run_metadata.json"
    cm_path = seed_dir / "confusion_matrix_test.csv"

    if not summary_path.exists() or not metadata_path.exists() or not cm_path.exists():
        return None

    summary = read_json(summary_path)
    metadata = read_json(metadata_path)
    cm = read_confusion_matrix(cm_path)
    cm_metrics = metrics_from_cm(cm)

    row = {
        "run_name": metadata["run_name"],
        "model_family": metadata["model_family"],
        "model": metadata["model"],
        "feature_set": metadata["feature_set"],
        "seed": metadata["seed"],
        "run_dir": str(seed_dir),
        "source_best_run_name": metadata["source_best_run_name"],
        "data": summary.get("data"),
        "best_epoch": summary.get("best_epoch"),
        "best_iteration": summary.get("best_iteration"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_acc": summary.get("best_val_acc"),
        "best_val_macro_f1": summary.get("best_val_macro_f1"),
        "test_loss": summary.get("test_loss"),
        "test_acc": summary.get("test_acc"),
        "test_macro_f1": summary.get("test_macro_f1"),
        "test_balanced_acc": summary.get("test_balanced_acc"),
        "total_train_seconds": summary.get("total_train_seconds"),
        **cm_metrics,
    }

    cm_long = confusion_matrix_to_long(
        cm,
        {
            "run_name": metadata["run_name"],
            "model_family": metadata["model_family"],
            "model": metadata["model"],
            "feature_set": metadata["feature_set"],
            "seed": metadata["seed"],
            "run_dir": str(seed_dir),
        },
    )
    return row, cm_long


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    combined_dir = output_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    cm_frames = []
    for family in MODEL_FAMILY_ORDER:
        family_dir = output_root / MODEL_FAMILY_DISPLAY[family]
        for seed_dir in sorted(family_dir.glob("seed_*")):
            collected = collect_seed_run(seed_dir)
            if collected is None:
                continue
            metric_row, cm_long = collected
            metric_rows.append(metric_row)
            cm_frames.append(cm_long)

    if not metric_rows:
        raise ValueError(f"No completed seed runs found under {output_root}")

    metrics_df = pd.DataFrame(metric_rows).sort_values(["model_family", "seed"])
    cm_df = pd.concat(cm_frames, ignore_index=True)

    summary_rows = []
    for family, group in metrics_df.groupby("model_family", sort=False):
        row = {
            "model_family": family,
            "model": group["model"].iloc[0],
            "feature_set": group["feature_set"].iloc[0],
            "completed_seeds": int(group["seed"].nunique()),
            "seeds": "|".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
        }
        for metric in ["overall_accuracy", "macro_f1", "weighted_f1", "test_acc", "test_macro_f1"]:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    write_csv(combined_dir / "all_run_metrics.csv", metrics_df)
    write_csv(combined_dir / "all_confusion_matrices_long.csv", cm_df)
    write_csv(combined_dir / "model_family_summary.csv", summary_df)

    manifest = {
        "output_root": str(output_root),
        "completed_runs": int(len(metrics_df)),
        "completed_model_families": sorted(metrics_df["model_family"].unique().tolist()),
        "combined_outputs": [
            "combined/all_run_metrics.csv",
            "combined/all_confusion_matrices_long.csv",
            "combined/model_family_summary.csv",
        ],
    }
    write_json(output_root / "experiment_manifest.json", manifest)

    print(f"Saved: {combined_dir / 'all_run_metrics.csv'}")
    print(f"Saved: {combined_dir / 'all_confusion_matrices_long.csv'}")
    print(f"Saved: {combined_dir / 'model_family_summary.csv'}")


if __name__ == "__main__":
    main()

