#!/usr/bin/env python3
"""
Train an XGBoost classifier on AE64 samples stored in NPZ.

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
- trains an XGBoost multiclass classifier
- evaluates on train/val across boosting iterations
- supports early stopping
- saves:
    * best model json file
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
python scripts/training/train_xgboost_ae64_from_npz.py \
  --data data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz \
  --outdir runs/xgb_ae64_lr0.03_ne5000_md10_mcw3_sub0.8_col0.8_es200_v1 \
  --learning-rate 0.03 \
  --n-estimators 5000 \
  --max-depth 10 \
  --min-child-weight 3.0 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --gamma 0.0 \
  --reg-alpha 0.0 \
  --reg-lambda 1.0 \
  --max-bin 256 \
  --early-stopping-rounds 200 \
  --eval-every 25 \
  --n-jobs 16 \
  --seed 42 \
  --tree-method hist \
  --save-test-preds

Notes:
- Labels are expected to be class IDs 1..10.
- Internally the model uses 0..9 for XGBoost.
- Expected input_dim is 64 for AE64-only features.
- Tree-based models do not require feature normalization, so mu/sigma are only validated
  for NPZ consistency and metadata compatibility.
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
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES_FIXED = 10


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost classifier on AE64 NPZ.")
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
        help="XGBoost learning rate / eta (default: 0.03).",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=5000,
        help="Maximum boosting rounds (default: 5000).",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum tree depth (default: 10).",
    )
    p.add_argument(
        "--min-child-weight",
        type=float,
        default=3.0,
        help="Minimum child weight (default: 3.0).",
    )
    p.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsampling ratio (default: 0.8).",
    )
    p.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Feature subsampling ratio per tree (default: 0.8).",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Minimum loss reduction required to make a split (default: 0.0).",
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
        default=1.0,
        help="L2 regularization (default: 1.0).",
    )
    p.add_argument(
        "--max-bin",
        type=int,
        default=256,
        help="Maximum number of histogram bins when using hist/approx tree methods (default: 256).",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=200,
        help="Early stopping rounds on validation mlogloss (default: 200).",
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
        "--tree-method",
        type=str,
        default="hist",
        choices=["auto", "exact", "approx", "hist"],
        help="XGBoost tree method (default: hist).",
    )
    p.add_argument(
        "--grow-policy",
        type=str,
        default="depthwise",
        choices=["depthwise", "lossguide"],
        help="Tree grow policy (default: depthwise).",
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


def flatten_eval_history(evals_result: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, float]]:
    datasets = list(evals_result.keys())

    n_rounds = 0
    for ds in datasets:
        for _, vals in evals_result[ds].items():
            n_rounds = max(n_rounds, len(vals))

    rows: List[Dict[str, float]] = []
    for i in range(n_rounds):
        row: Dict[str, float] = {"iteration": i + 1}
        for ds in datasets:
            for metric_name, vals in evals_result[ds].items():
                row[f"{ds}_{metric_name}"] = float(vals[i]) if i < len(vals) else math.nan
        rows.append(row)
    return rows


def get_best_iteration(booster: xgb.Booster, fallback: int) -> int:
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is None:
        return fallback
    return int(best_iter) + 1


def predict_proba_best(booster: xgb.Booster, X: np.ndarray, best_iteration: int) -> np.ndarray:
    dmat = xgb.DMatrix(X)
    proba = booster.predict(dmat, iteration_range=(0, best_iteration))
    proba = np.asarray(proba, dtype=np.float32)
    if proba.ndim != 2 or proba.shape[1] != NUM_CLASSES_FIXED:
        raise SystemExit(f"Unexpected prediction shape: {proba.shape}")
    return proba


def save_feature_importance_csv(path: Path, booster: xgb.Booster, feature_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    gain_map = booster.get_score(importance_type="gain")
    rows = []
    for name in feature_names:
        rows.append({
            "feature_name": name,
            "importance_gain": float(gain_map.get(name, 0.0)),
        })
    rows.sort(key=lambda x: x["importance_gain"], reverse=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_name", "importance_gain"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.data = resolve_path(args.data)
    args.outdir = resolve_path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

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

    log(f"Train samples       : {X_train.shape[0]}")
    log(f"Val samples         : {X_val.shape[0]}")
    log(f"Test samples        : {X_test.shape[0] if has_test and X_test is not None else 0}")
    log(f"Input dim           : {input_dim}")
    log(f"Num classes         : {num_classes}")
    log(f"Learning rate       : {args.learning_rate}")
    log(f"n_estimators        : {args.n_estimators}")
    log(f"max_depth           : {args.max_depth}")
    log(f"min_child_weight    : {args.min_child_weight}")
    log(f"subsample           : {args.subsample}")
    log(f"colsample_bytree    : {args.colsample_bytree}")
    log(f"gamma               : {args.gamma}")
    log(f"reg_alpha           : {args.reg_alpha}")
    log(f"reg_lambda          : {args.reg_lambda}")
    log(f"max_bin             : {args.max_bin}")
    log(f"early_stopping      : {args.early_stopping_rounds}")
    log(f"tree_method         : {args.tree_method}")
    log(f"grow_policy         : {args.grow_policy}")
    log(f"n_jobs              : {args.n_jobs}")
    log(f"Feature tail        : {feature_names[-10:].tolist()}")
    log(f"Class weights       : {class_weights_np.tolist()}")

    feature_names_list = [str(x) for x in feature_names.tolist()]

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train_zero,
        weight=sample_weights_train,
        feature_names=feature_names_list,
    )
    dval = xgb.DMatrix(
        X_val,
        label=y_val_zero,
        weight=sample_weights_val,
        feature_names=feature_names_list,
    )

    params: Dict[str, object] = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": ["mlogloss", "merror"],
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "alpha": args.reg_alpha,
        "lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "tree_method": args.tree_method,
        "grow_policy": args.grow_policy,
        "verbosity": 1,
        "seed": args.seed,
        "nthread": args.n_jobs,
    }

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    evals = [(dtrain, "train"), (dval, "val")]

    log("Starting XGBoost training.")
    train_start = time.time()

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.n_estimators,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.eval_every,
    )

    total_seconds = time.time() - train_start
    best_iteration = get_best_iteration(booster, args.n_estimators)
    best_score = getattr(booster, "best_score", None)

    log(f"Training finished in {total_seconds:.1f}s")
    log(f"Best iteration: {best_iteration}")

    model_json_path = args.outdir / "best_model.json"
    summary_json_path = args.outdir / "summary.json"
    history_csv_path = args.outdir / "history.csv"

    cm_val_csv_path = args.outdir / "confusion_matrix_val.csv"
    val_preds_csv_path = args.outdir / "val_predictions.csv"
    per_class_val_csv_path = args.outdir / "per_class_metrics_val.csv"

    cm_test_csv_path = args.outdir / "confusion_matrix_test.csv"
    test_preds_csv_path = args.outdir / "test_predictions.csv"
    per_class_test_csv_path = args.outdir / "per_class_metrics_test.csv"
    feature_importance_csv_path = args.outdir / "feature_importance_gain.csv"

    booster.save_model(str(model_json_path))
    save_history_csv(history_csv_path, flatten_eval_history(evals_result))

    log("Evaluating best model on train/val.")
    y_train_proba = predict_proba_best(booster, X_train, best_iteration)
    y_val_proba = predict_proba_best(booster, X_val, best_iteration)

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
        y_test_proba = predict_proba_best(booster, X_test, best_iteration)
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

    save_feature_importance_csv(feature_importance_csv_path, booster, feature_names_list)

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
        "seed": args.seed,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]) if has_test and X_test is not None else 0,
        "learning_rate": args.learning_rate,
        "n_estimators_requested": args.n_estimators,
        "best_iteration": best_iteration,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "early_stopping_rounds": args.early_stopping_rounds,
        "eval_every": args.eval_every,
        "n_jobs": args.n_jobs,
        "tree_method": args.tree_method,
        "grow_policy": args.grow_policy,
        "class_weights": class_weights_np.tolist(),
        "best_score_from_xgboost": best_score,
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
        "best_model_path": str(model_json_path),
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

    log(
        f"Train metrics         : "
        f"loss={best_train_loss:.5f} acc={best_train_acc:.4f} "
        f"macro_f1={best_train_macro_f1:.4f} bal_acc={best_train_bal_acc:.4f}"
    )
    log(
        f"Best val metrics      : "
        f"loss={best_val_loss:.5f} acc={best_val_acc:.4f} "
        f"macro_f1={best_val_macro_f1:.4f} bal_acc={best_val_bal_acc:.4f}"
    )
    if test_done:
        log(f"Test macro F1         : {test_macro_f1:.4f}")
        log(f"Test balanced acc     : {test_bal_acc:.4f}")
    log(f"Saved summary         : {summary_json_path}")
    log(f"Saved history         : {history_csv_path}")
    log(f"Saved best model      : {model_json_path}")
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
