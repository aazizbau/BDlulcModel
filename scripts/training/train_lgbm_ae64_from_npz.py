#!/usr/bin/env python3
"""
Train a LightGBM classifier on AE64 samples stored in NPZ.

Expected NPZ keys:
  Required:
    X_train, y_train, X_val, y_val, mu, sigma, feature_names, meta

  Optional:
    X_test, y_test

This script:
- loads NPZ
- checks labels are in 1..10
- converts labels from 1..10 to 0..9 internally
- computes class weights from y_train
- trains a LightGBM multiclass classifier
- evaluates on train/val across boosting iterations
- supports early stopping
- saves:
    * best model text file
    * summary.json
    * per-iteration history.csv
    * confusion_matrix_val.csv
    * val_predictions.csv
    * per_class_metrics_val.csv
    * if test exists:
        - confusion_matrix_test.csv
        - test_predictions.csv
        - per_class_metrics_test.csv

Example:
python scripts/training/train_lgbm_ae64_from_npz.py \
  --data data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz \
  --outdir runs/lgbm_ae64_lr0.03_ne5000_nl127_mcs50_sub0.8_col0.8_es200_v1 \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --num-leaves 127 \
  --min-child-samples 50 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-alpha 0.0 \
  --reg-lambda 0.0 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --force-col-wise \
  --save-test-preds

Notes:
- Labels are expected to be class IDs 1..10.
- Internally the model uses 0..9 for LightGBM.
- Expected input_dim is 64 for AE64-only features.
- Tree-based models do not require feature normalization, so mu/sigma are only validated
  for NPZ consistency and metadata compatibility.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Extract spatially split samples, train a classifier, or orchestrate hyperparameter experiments.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--data``, ``--outdir``, ``--learning-rate``, ``--n-estimators``, ``--num-leaves``, ``--max-depth``, ``--min-child-samples``, ``--subsample``, ``--subsample-freq``, ``--colsample-bytree``, ``--min-split-gain``, ``--reg-alpha``, ``--reg-lambda``, ``--max-bin``, ``--early-stopping-rounds``, ``--eval-every``, ``--n-jobs``, ``--seed``, ``--force-col-wise``, ``--force-row-wise``, ``--deterministic``, ``--save-test-preds``. Run the ``--help`` command below for required values, defaults, and accepted choices.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replace NPZ/raster/vector inputs with samples extracted from the new AOI, preserve spatially disjoint splits, and review class IDs, feature order, block size, budgets, and random seeds.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES_FIXED = 10
EXPECTED_INPUT_DIM = 64


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM classifier on AE64 NPZ.")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz"),
        help="Input NPZ file containing X_train/X_val/y_train/y_val/mu/sigma and optionally X_test/y_test.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output run directory.",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="LightGBM learning rate (default: 0.03).",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=5000,
        help="Maximum boosting rounds (default: 5000).",
    )
    p.add_argument(
        "--num-leaves",
        type=int,
        default=127,
        help="Maximum number of leaves (default: 127).",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=-1,
        help="Maximum tree depth; -1 means no limit (default: -1).",
    )
    p.add_argument(
        "--min-child-samples",
        type=int,
        default=50,
        help="Minimum samples per leaf (default: 50).",
    )
    p.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsampling ratio (default: 0.8).",
    )
    p.add_argument(
        "--subsample-freq",
        type=int,
        default=1,
        help="Frequency for row subsampling (default: 1).",
    )
    p.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Feature subsampling ratio per tree (default: 0.8).",
    )
    p.add_argument(
        "--min-split-gain",
        type=float,
        default=0.0,
        help="Minimum gain to perform a split (default: 0.0).",
    )
    p.add_argument(
        "--reg-alpha",
        type=float,
        default=0.0,
        help="L1 regularization (default: 0.0).",
    )
    p.add_argument(
        "--reg-lambda",
        type=float,
        default=0.0,
        help="L2 regularization (default: 0.0).",
    )
    p.add_argument(
        "--max-bin",
        type=int,
        default=255,
        help="Maximum number of bins (default: 255).",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=200,
        help="Early stopping rounds on validation multi_logloss (default: 200).",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="Log/eval period during training (default: 25).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of CPU threads (default: 8).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--force-col-wise",
        action="store_true",
        help="Force LightGBM column-wise histogram construction.",
    )
    p.add_argument(
        "--force-row-wise",
        action="store_true",
        help="Force LightGBM row-wise histogram construction.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode in LightGBM.",
    )
    p.add_argument(
        "--save-test-preds",
        action="store_true",
        help="Save test_predictions.csv if test split exists.",
    )
    return p.parse_args()


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def balanced_acc_from_cm(cm: np.ndarray) -> float:
    recalls = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        recalls.append(safe_div(tp, tp + fn))
    return float(np.mean(recalls))


def accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0


def multi_logloss_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if y_true.size == 0:
        return math.nan
    eps = 1e-15
    p = np.clip(y_proba[np.arange(y_true.shape[0]), y_true], eps, 1.0 - eps)
    return float(-np.mean(np.log(p)))


def compute_class_weights(y_train_zero_based: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y_train_zero_based, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.zeros_like(counts, dtype=np.float64)

    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total / (num_classes * counts[i])
        else:
            weights[i] = 0.0

    nonzero = weights[weights > 0]
    mean_nonzero = float(nonzero.mean()) if nonzero.size > 0 else 0.0
    if mean_nonzero > 0:
        weights = weights / mean_nonzero

    return weights.astype(np.float32)


def ensure_finite_array(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise SystemExit(f"Non-finite values found in {name}.")


def validate_feature_stats(mu: np.ndarray, sigma: np.ndarray, input_dim: int) -> None:
    if mu.ndim != 1 or sigma.ndim != 1:
        raise SystemExit("mu and sigma must both be 1D arrays.")
    if len(mu) != input_dim or len(sigma) != input_dim:
        raise SystemExit(
            f"mu/sigma length mismatch with input_dim: len(mu)={len(mu)} len(sigma)={len(sigma)} input_dim={input_dim}"
        )


def validate_labels(name: str, y: np.ndarray) -> None:
    if y.ndim != 1:
        raise SystemExit(f"{name} must be 1D.")
    unique = sorted(np.unique(y).tolist())
    allowed = set(range(1, NUM_CLASSES_FIXED + 1))
    if not set(unique).issubset(allowed):
        raise SystemExit(f"{name} contains invalid labels. Found {unique}, expected subset of 1..{NUM_CLASSES_FIXED}.")


def describe_label_presence(y: np.ndarray) -> Dict[str, List[int]]:
    present = sorted(np.unique(y).tolist())
    missing = [i for i in range(1, NUM_CLASSES_FIXED + 1) if i not in present]
    return {"present": present, "missing": missing}


def per_class_metrics_from_cm(cm: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "class_id": c + 1,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    return rows


def save_history_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_confusion_matrix_csv(path: Path, cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["true\\pred"] + [str(i) for i in range(1, cm.shape[0] + 1)]
        writer.writerow(header)
        for i in range(cm.shape[0]):
            writer.writerow([str(i + 1)] + cm[i].tolist())


def save_predictions_csv(path: Path, y_true_zero: np.ndarray, y_pred_zero: np.ndarray, y_proba: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        header = ["y_true", "y_pred"]
        if y_proba is not None:
            header.extend([f"prob_class_{i}" for i in range(1, y_proba.shape[1] + 1)])
        writer.writerow(header)

        for i in range(len(y_true_zero)):
            row = [int(y_true_zero[i]) + 1, int(y_pred_zero[i]) + 1]
            if y_proba is not None:
                row.extend([float(x) for x in y_proba[i]])
            writer.writerow(row)


def save_per_class_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def evaluate_from_proba(
    y_true_zero: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    y_pred_zero = np.argmax(y_proba, axis=1).astype(np.int64)
    cm = confusion_matrix_np(y_true_zero, y_pred_zero, num_classes)
    return {
        "loss": multi_logloss_from_proba(y_true_zero, y_proba),
        "acc": accuracy_np(y_true_zero, y_pred_zero),
        "macro_f1": macro_f1_from_cm(cm),
        "balanced_acc": balanced_acc_from_cm(cm),
        "cm": cm,
        "y_pred": y_pred_zero,
    }


def make_sample_weights(y_zero: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    return class_weights[y_zero].astype(np.float32)


def flatten_eval_history(eval_results: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, float]]:
    datasets = list(eval_results.keys())
    metric_names: List[str] = []
    for ds in datasets:
        for m in eval_results[ds].keys():
            key = f"{ds}_{m}"
            if key not in metric_names:
                metric_names.append(key)

    n_rounds = 0
    for ds in datasets:
        for m, vals in eval_results[ds].items():
            n_rounds = max(n_rounds, len(vals))

    rows: List[Dict[str, float]] = []
    for i in range(n_rounds):
        row: Dict[str, float] = {"iteration": i + 1}
        for ds in datasets:
            for m, vals in eval_results[ds].items():
                row[f"{ds}_{m}"] = float(vals[i]) if i < len(vals) else math.nan
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.data = resolve_path(args.data)
    args.outdir = resolve_path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.force_col_wise and args.force_row_wise:
        raise SystemExit("Use only one of --force-col-wise or --force-row-wise, not both.")

    args_serializable = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_serializable[k] = str(v)
        else:
            args_serializable[k] = v

    if not args.data.exists():
        raise SystemExit(f"Input NPZ not found: {args.data}")

    set_seed(args.seed)

    log(f"Loading NPZ: {args.data}")
    with np.load(args.data, allow_pickle=True) as d:
        required_keys = ["X_train", "y_train", "X_val", "y_val", "mu", "sigma", "feature_names", "meta"]
        missing_required = [k for k in required_keys if k not in d]
        if missing_required:
            raise SystemExit(f"Missing required NPZ keys: {missing_required}")

        X_train = d["X_train"].astype(np.float32)
        y_train = d["y_train"].astype(np.int64)
        X_val = d["X_val"].astype(np.float32)
        y_val = d["y_val"].astype(np.int64)
        mu = d["mu"].astype(np.float32)
        sigma = d["sigma"].astype(np.float32)
        feature_names = d["feature_names"]
        meta = json.loads(str(d["meta"]))

        has_test = ("X_test" in d) and ("y_test" in d)
        X_test = d["X_test"].astype(np.float32) if has_test else None
        y_test = d["y_test"].astype(np.int64) if has_test else None

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise SystemExit("X_train and X_val must be 2D.")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise SystemExit("y_train and y_val must be 1D.")
    if X_train.shape[1] != X_val.shape[1]:
        raise SystemExit("X_train and X_val must have same number of columns.")
    if X_train.shape[0] != y_train.shape[0]:
        raise SystemExit("X_train and y_train row count mismatch.")
    if X_val.shape[0] != y_val.shape[0]:
        raise SystemExit("X_val and y_val row count mismatch.")

    if has_test:
        if X_test is None or y_test is None:
            raise SystemExit("Internal error: has_test inconsistent.")
        if X_test.ndim != 2 or y_test.ndim != 1:
            raise SystemExit("X_test must be 2D and y_test must be 1D.")
        if X_test.shape[1] != X_train.shape[1]:
            raise SystemExit("X_test must have same number of columns as X_train.")
        if X_test.shape[0] != y_test.shape[0]:
            raise SystemExit("X_test and y_test row count mismatch.")

    input_dim = int(X_train.shape[1])
    validate_feature_stats(mu, sigma, input_dim)

    if input_dim != 64:
        log(f"Warning: expected 64 features for AE64-only, found {input_dim}")

    validate_labels("y_train", y_train)
    validate_labels("y_val", y_val)
    if has_test and y_test is not None:
        validate_labels("y_test", y_test)

    ensure_finite_array("X_train", X_train)
    ensure_finite_array("X_val", X_val)
    if has_test and X_test is not None:
        ensure_finite_array("X_test", X_test)

    train_presence = describe_label_presence(y_train)
    val_presence = describe_label_presence(y_val)
    test_presence = describe_label_presence(y_test) if has_test and y_test is not None else None

    log(f"Train label presence: present={train_presence['present']} missing={train_presence['missing']}")
    log(f"Val label presence  : present={val_presence['present']} missing={val_presence['missing']}")
    if test_presence is not None:
        log(f"Test label presence : present={test_presence['present']} missing={test_presence['missing']}")

    y_train_zero = y_train - 1
    y_val_zero = y_val - 1
    y_test_zero = (y_test - 1) if has_test and y_test is not None else None

    num_classes = NUM_CLASSES_FIXED
    class_weights_np = compute_class_weights(y_train_zero, num_classes=num_classes)
    sample_weights_train = make_sample_weights(y_train_zero, class_weights_np)
    sample_weights_val = make_sample_weights(y_val_zero, class_weights_np)

    log(f"Train samples     : {X_train.shape[0]}")
    log(f"Val samples       : {X_val.shape[0]}")
    log(f"Test samples      : {X_test.shape[0] if has_test and X_test is not None else 0}")
    log(f"Input dim         : {input_dim}")
    log(f"Num classes       : {num_classes}")
    log(f"Learning rate     : {args.learning_rate}")
    log(f"n_estimators      : {args.n_estimators}")
    log(f"num_leaves        : {args.num_leaves}")
    log(f"max_depth         : {args.max_depth}")
    log(f"min_child_samples : {args.min_child_samples}")
    log(f"subsample         : {args.subsample}")
    log(f"colsample_bytree  : {args.colsample_bytree}")
    log(f"reg_alpha         : {args.reg_alpha}")
    log(f"reg_lambda        : {args.reg_lambda}")
    log(f"max_bin           : {args.max_bin}")
    log(f"early_stopping    : {args.early_stopping_rounds}")
    log(f"n_jobs            : {args.n_jobs}")
    log(f"Feature tail      : {feature_names[-10:].tolist()}")
    log(f"Class weights     : {class_weights_np.tolist()}")

    train_set = lgb.Dataset(
        X_train,
        label=y_train_zero,
        weight=sample_weights_train,
        feature_name=[str(x) for x in feature_names.tolist()],
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val_zero,
        weight=sample_weights_val,
        reference=train_set,
        feature_name=[str(x) for x in feature_names.tolist()],
        free_raw_data=False,
    )

    params: Dict[str, object] = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": ["multi_logloss", "multi_error"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "colsample_bytree": args.colsample_bytree,
        "min_split_gain": args.min_split_gain,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "verbosity": -1,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "data_random_seed": args.seed,
        "num_threads": args.n_jobs,
    }

    if args.force_col_wise:
        params["force_col_wise"] = True
    if args.force_row_wise:
        params["force_row_wise"] = True
    if args.deterministic:
        params["deterministic"] = True

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    history_rows_runtime: List[Dict[str, float]] = []

    def _record_callback(env: lgb.callback.CallbackEnv) -> None:
        row: Dict[str, float] = {"iteration": int(env.iteration) + 1}
        for item in env.evaluation_result_list:
            dataset_name = str(item[0])
            metric_name = str(item[1])
            metric_value = float(item[2])
            row[f"{dataset_name}_{metric_name}"] = metric_value
        history_rows_runtime.append(row)

    callbacks = [
        lgb.record_evaluation(evals_result),
        lgb.log_evaluation(period=args.eval_every),
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
        _record_callback,
    ]

    log("Starting LightGBM training.")
    train_start = time.time()

    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=args.n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    total_seconds = time.time() - train_start
    best_iteration = int(booster.best_iteration or args.n_estimators)
    best_score = booster.best_score

    log(f"Training finished in {total_seconds:.1f}s")
    log(f"Best iteration: {best_iteration}")

    model_txt_path = args.outdir / "best_model.txt"
    summary_json_path = args.outdir / "summary.json"
    history_csv_path = args.outdir / "history.csv"

    cm_val_csv_path = args.outdir / "confusion_matrix_val.csv"
    val_preds_csv_path = args.outdir / "val_predictions.csv"
    per_class_val_csv_path = args.outdir / "per_class_metrics_val.csv"

    cm_test_csv_path = args.outdir / "confusion_matrix_test.csv"
    test_preds_csv_path = args.outdir / "test_predictions.csv"
    per_class_test_csv_path = args.outdir / "per_class_metrics_test.csv"
    feature_importance_csv_path = args.outdir / "feature_importance_gain.csv"

    booster.save_model(str(model_txt_path), num_iteration=best_iteration)

    if history_rows_runtime:
        all_keys = ["iteration"]
        for row in history_rows_runtime:
            for k in row.keys():
                if k not in all_keys:
                    all_keys.append(k)
        normalized_rows: List[Dict[str, float]] = []
        for row in history_rows_runtime:
            nr = {k: row.get(k, math.nan) for k in all_keys}
            normalized_rows.append(nr)
        save_history_csv(history_csv_path, normalized_rows)
    else:
        save_history_csv(history_csv_path, flatten_eval_history(evals_result))

    log("Evaluating best model on train/val.")
    y_train_proba = booster.predict(X_train, num_iteration=best_iteration)
    y_val_proba = booster.predict(X_val, num_iteration=best_iteration)

    train_metrics = evaluate_from_proba(y_train_zero, y_train_proba, num_classes=num_classes)
    val_metrics = evaluate_from_proba(y_val_zero, y_val_proba, num_classes=num_classes)

    save_confusion_matrix_csv(cm_val_csv_path, val_metrics["cm"])
    save_predictions_csv(val_preds_csv_path, y_val_zero, val_metrics["y_pred"], y_val_proba)
    save_per_class_metrics_csv(
        per_class_val_csv_path,
        per_class_metrics_from_cm(val_metrics["cm"]),
    )

    test_done = False
    test_loss = math.nan
    test_acc = math.nan
    test_macro_f1 = math.nan
    test_bal_acc = math.nan

    if has_test and X_test is not None and y_test_zero is not None:
        log("Evaluating best model on test split.")
        y_test_proba = booster.predict(X_test, num_iteration=best_iteration)
        test_metrics = evaluate_from_proba(y_test_zero, y_test_proba, num_classes=num_classes)

        test_loss = float(test_metrics["loss"])
        test_acc = float(test_metrics["acc"])
        test_macro_f1 = float(test_metrics["macro_f1"])
        test_bal_acc = float(test_metrics["balanced_acc"])

        save_confusion_matrix_csv(cm_test_csv_path, test_metrics["cm"])
        if args.save_test_preds:
            save_predictions_csv(test_preds_csv_path, y_test_zero, test_metrics["y_pred"], y_test_proba)
        save_per_class_metrics_csv(
            per_class_test_csv_path,
            per_class_metrics_from_cm(test_metrics["cm"]),
        )

        log(
            f"Test metrics | "
            f"loss={test_loss:.5f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f} bal_acc={test_bal_acc:.4f}"
        )
        test_done = True
    else:
        log("No test split found in NPZ. Skipping test evaluation.")

    feature_names_list = booster.feature_name()
    importances_gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(importances_gain)[::-1]
    with feature_importance_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "importance_gain"])
        for idx in order:
            writer.writerow([feature_names_list[idx], float(importances_gain[idx])])

    best_train_loss = float(train_metrics["loss"])
    best_train_acc = float(train_metrics["acc"])
    best_train_macro_f1 = float(train_metrics["macro_f1"])
    best_train_bal_acc = float(train_metrics["balanced_acc"])

    best_val_loss = float(val_metrics["loss"])
    best_val_acc = float(val_metrics["acc"])
    best_val_macro_f1 = float(val_metrics["macro_f1"])
    best_val_bal_acc = float(val_metrics["balanced_acc"])

    summary = {
        "created_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "data": str(args.data),
        "outdir": str(args.outdir),
        "model": "LightGBM",
        "model_type": "lgbm_classifier",
        "seed": args.seed,
        "input_dim": input_dim,
        "expected_input_dim": EXPECTED_INPUT_DIM,
        "num_classes": num_classes,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]) if has_test and X_test is not None else 0,
        "learning_rate": args.learning_rate,
        "n_estimators_requested": args.n_estimators,
        "best_iteration": best_iteration,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "colsample_bytree": args.colsample_bytree,
        "min_split_gain": args.min_split_gain,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "early_stopping_rounds": args.early_stopping_rounds,
        "eval_every": args.eval_every,
        "n_jobs": args.n_jobs,
        "force_col_wise": args.force_col_wise,
        "force_row_wise": args.force_row_wise,
        "deterministic": args.deterministic,
        "class_weights": class_weights_np.tolist(),
        "best_scores_from_lgbm": best_score,
        "train_loss": best_train_loss,
        "train_acc": best_train_acc,
        "train_macro_f1": best_train_macro_f1,
        "train_balanced_acc": best_train_bal_acc,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_balanced_acc": best_val_bal_acc,
        "test_evaluated": test_done,
        "test_loss": float(test_loss) if np.isfinite(test_loss) else None,
        "test_acc": float(test_acc) if np.isfinite(test_acc) else None,
        "test_macro_f1": float(test_macro_f1) if np.isfinite(test_macro_f1) else None,
        "test_balanced_acc": float(test_bal_acc) if np.isfinite(test_bal_acc) else None,
        "total_train_seconds": float(total_seconds),
        "history_csv": str(history_csv_path),
        "best_model_path": str(model_txt_path),
        "feature_importance_gain_csv": str(feature_importance_csv_path),
        "confusion_matrix_val_csv": str(cm_val_csv_path),
        "val_predictions_csv": str(val_preds_csv_path),
        "per_class_metrics_val_csv": str(per_class_val_csv_path),
        "confusion_matrix_test_csv": str(cm_test_csv_path) if test_done else None,
        "test_predictions_csv": str(test_preds_csv_path) if (test_done and args.save_test_preds) else None,
        "per_class_metrics_test_csv": str(per_class_test_csv_path) if test_done else None,
        "train_label_presence": train_presence,
        "val_label_presence": val_presence,
        "test_label_presence": test_presence,
        "data_meta": meta,
        "args": args_serializable,
    }

    write_json(summary_json_path, summary)

    log(f"Train metrics         : loss={best_train_loss:.5f} acc={best_train_acc:.4f} macro_f1={best_train_macro_f1:.4f} bal_acc={best_train_bal_acc:.4f}")
    log(f"Best val metrics      : loss={best_val_loss:.5f} acc={best_val_acc:.4f} macro_f1={best_val_macro_f1:.4f} bal_acc={best_val_bal_acc:.4f}")
    if test_done:
        log(f"Test macro F1         : {test_macro_f1:.4f}")
        log(f"Test balanced acc     : {test_bal_acc:.4f}")
    log(f"Saved summary         : {summary_json_path}")
    log(f"Saved history         : {history_csv_path}")
    log(f"Saved best model      : {model_txt_path}")
    log(f"Saved feature import. : {feature_importance_csv_path}")
    log(f"Saved val confusion   : {cm_val_csv_path}")
    log(f"Saved val predictions : {val_preds_csv_path}")
    log(f"Saved val per-class   : {per_class_val_csv_path}")
    if test_done:
        log(f"Saved test confusion  : {cm_test_csv_path}")
        if args.save_test_preds:
            log(f"Saved test predictions: {test_preds_csv_path}")
        log(f"Saved test per-class  : {per_class_test_csv_path}")


if __name__ == "__main__":
    main()
