#!/usr/bin/env python3
"""
Train an MLP classifier on AE64 + 10 indices samples stored in NPZ.

Expected NPZ keys:
  Required:
    X_train, y_train, X_val, y_val, mu, sigma, feature_names, meta

  Optional:
    X_test, y_test

This script:
- loads NPZ
- normalizes X using mu/sigma from the NPZ
- enforces labels to be in 1..10
- converts labels from 1..10 to 0..9 internally
- computes class weights from y_train
- trains an MLP classifier with weighted cross-entropy
- evaluates train/val each epoch
- supports early stopping
- optionally uses ReduceLROnPlateau
- saves:
    * best model checkpoint
    * summary.json
    * per-epoch history.csv
    * confusion_matrix_val.csv
    * val_predictions.csv
    * per_class_metrics_val.csv
    * if test exists:
        - confusion_matrix_test.csv
        - test_predictions.csv
        - per_class_metrics_test.csv

Example:
python scripts/training/train_mlp_ae64_plus10indices_from_npz.py \
  --data data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz \
  --outdir runs/mlp_ae64plus10idx_h512-256_do02_lr1e3_bs4096_v1 \
  --hidden 512 256 \
  --dropout 0.2 \
  --batch-size 4096 \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 15 \
  --min-delta 1e-4 \
  --label-smoothing 0.05 \
  --scheduler \
  --scheduler-factor 0.5 \
  --scheduler-patience 5 \
  --eval-every 1 \
  --device cuda \
  --seed 42

Notes:
- Labels are expected to be class IDs 1..10.
- Internally the model uses 0..9 for PyTorch cross-entropy.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Extract spatially split samples, train a classifier, or orchestrate hyperparameter experiments.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line interface exposes ``--data``, ``--outdir``, ``--hidden``, ``--dropout``, ``--batch-size``, ``--epochs``, ``--lr``, ``--weight-decay``, ``--patience``, ``--min-delta``, ``--label-smoothing``, ``--scheduler``, ``--scheduler-factor``, ``--scheduler-patience``, ``--eval-every``, ``--device``, ``--seed``. Run the ``--help`` command below for required values, defaults, and accepted choices.
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
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


JST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES_FIXED = 10
EXPECTED_INPUT_DIM = 74


def log(message: str) -> None:
    ts = datetime.now(JST).isoformat(timespec="seconds")
    print(f"[{ts}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP classifier on AE64 + 10 indices NPZ.")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz"),
        help="Input NPZ file containing X_train/X_val/y_train/y_val/mu/sigma and optionally X_test/y_test.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output run directory.",
    )
    p.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer sizes, e.g. --hidden 512 256",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (default: 0.2).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size (default: 8192).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum epochs (default: 100).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience on validation macro F1 (default: 15).",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement required for early stopping (default: 1e-4).",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for CrossEntropyLoss (default: 0.0).",
    )
    p.add_argument(
        "--scheduler",
        action="store_true",
        help="Enable ReduceLROnPlateau scheduler on validation loss.",
    )
    p.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor (default: 0.5).",
    )
    p.add_argument(
        "--scheduler-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (default: 5).",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate every N epochs (default: 1).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: auto).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return p.parse_args()


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EpochMetrics:
    epoch: int
    lr: float
    train_loss: float
    train_acc: float
    train_macro_f1: float
    val_loss: float
    val_acc: float
    val_macro_f1: float
    val_balanced_acc: float
    epoch_seconds: float


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


def normalize_features(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma_safe = np.where(sigma == 0, 1.0, sigma).astype(np.float32)
    Xn = (X - mu) / sigma_safe
    return Xn.astype(np.float32)


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_n = 0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_proba: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            n = xb.size(0)
            total_loss += float(loss.item()) * n
            total_n += n

            pred = torch.argmax(logits, dim=1)
            proba = torch.softmax(logits, dim=1)
            all_true.append(yb.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_proba.append(proba.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.zeros((0,), dtype=np.int64)
    y_proba = np.concatenate(all_proba) if all_proba else np.zeros((0, num_classes), dtype=np.float32)

    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    loss_avg = total_loss / total_n if total_n > 0 else math.nan
    acc = accuracy_np(y_true, y_pred)
    macro_f1 = macro_f1_from_cm(cm)
    bal_acc = balanced_acc_from_cm(cm)
    return loss_avg, acc, macro_f1, bal_acc, y_true, y_pred, y_proba


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


def save_history_csv(path: Path, history: List[EpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "train_macro_f1",
                "val_loss",
                "val_acc",
                "val_macro_f1",
                "val_balanced_acc",
                "epoch_seconds",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(asdict(row))


def save_confusion_matrix_csv(path: Path, cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["true\\pred"] + [str(i) for i in range(1, cm.shape[0] + 1)]
        writer.writerow(header)
        for i in range(cm.shape[0]):
            writer.writerow([str(i + 1)] + cm[i].tolist())


def save_predictions_csv(
    path: Path,
    y_true_zero: np.ndarray,
    y_pred_zero: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["y_true", "y_pred"]
        if y_proba is not None:
            header.extend([f"prob_class_{i}" for i in range(1, y_proba.shape[1] + 1)])
        writer.writerow(header)
        for i, (t, p) in enumerate(zip(y_true_zero, y_pred_zero)):
            row = [int(t) + 1, int(p) + 1]
            if y_proba is not None:
                row.extend(float(x) for x in y_proba[i])
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


def load_best_checkpoint_for_eval(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLPClassifier(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        num_classes=int(ckpt["num_classes"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


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
    device = torch.device(args.device)

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
    if has_test:
        if X_test is None or y_test is None:
            raise SystemExit("Internal error: has_test inconsistent.")
        if X_test.ndim != 2 or y_test.ndim != 1:
            raise SystemExit("X_test must be 2D and y_test must be 1D.")
        if X_test.shape[1] != X_train.shape[1]:
            raise SystemExit("X_test must have same number of columns as X_train.")

    input_dim = int(X_train.shape[1])
    validate_feature_stats(mu, sigma, input_dim)

    if input_dim != 74:
        log(f"Warning: expected 74 features for AE64 + 10 indices, found {input_dim}")

    validate_labels("y_train", y_train)
    validate_labels("y_val", y_val)
    if has_test and y_test is not None:
        validate_labels("y_test", y_test)

    train_presence = describe_label_presence(y_train)
    val_presence = describe_label_presence(y_val)
    test_presence = describe_label_presence(y_test) if has_test and y_test is not None else None

    log(f"Train label presence: present={train_presence['present']} missing={train_presence['missing']}")
    log(f"Val label presence  : present={val_presence['present']} missing={val_presence['missing']}")
    if test_presence is not None:
        log(f"Test label presence : present={test_presence['present']} missing={test_presence['missing']}")

    num_classes = NUM_CLASSES_FIXED

    y_train_zero = y_train - 1
    y_val_zero = y_val - 1
    y_test_zero = (y_test - 1) if has_test and y_test is not None else None

    log("Normalizing train/val/test using NPZ mu/sigma.")
    X_train = normalize_features(X_train, mu, sigma)
    X_val = normalize_features(X_val, mu, sigma)
    if has_test and X_test is not None:
        X_test = normalize_features(X_test, mu, sigma)

    ensure_finite_array("X_train", X_train)
    ensure_finite_array("X_val", X_val)
    if has_test and X_test is not None:
        ensure_finite_array("X_test", X_test)

    class_weights_np = compute_class_weights(y_train_zero, num_classes=num_classes)
    class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    log(f"Train samples     : {X_train.shape[0]}")
    log(f"Val samples       : {X_val.shape[0]}")
    log(f"Test samples      : {X_test.shape[0] if has_test and X_test is not None else 0}")
    log(f"Input dim         : {input_dim}")
    log(f"Num classes       : {num_classes}")
    log(f"Hidden dims       : {args.hidden}")
    log(f"Dropout           : {args.dropout}")
    log(f"Batch size        : {args.batch_size}")
    log(f"Epochs            : {args.epochs}")
    log(f"Learning rate     : {args.lr}")
    log(f"Weight decay      : {args.weight_decay}")
    log(f"Patience          : {args.patience}")
    log(f"Device            : {device}")
    log(f"Feature tail      : {feature_names[-10:].tolist()}")
    log(f"Class weights     : {class_weights_np.tolist()}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train_zero.astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val_zero.astype(np.int64)),
    )
    test_ds = (
        TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test_zero.astype(np.int64)),
        )
        if has_test and X_test is not None and y_test_zero is not None
        else None
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if test_ds is not None
        else None
    )

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_val_loss = math.inf
    best_val_acc = math.nan
    best_val_bal_acc = math.nan
    epochs_without_improve = 0
    history: List[EpochMetrics] = []

    best_ckpt_path = args.outdir / "best_model.pt"
    history_csv_path = args.outdir / "history.csv"
    summary_json_path = args.outdir / "summary.json"

    cm_val_csv_path = args.outdir / "confusion_matrix_val.csv"
    val_preds_csv_path = args.outdir / "val_predictions.csv"
    per_class_val_csv_path = args.outdir / "per_class_metrics_val.csv"

    cm_test_csv_path = args.outdir / "confusion_matrix_test.csv"
    test_preds_csv_path = args.outdir / "test_predictions.csv"
    per_class_test_csv_path = args.outdir / "per_class_metrics_test.csv"

    overall_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)

        if should_eval:
            train_loss, train_acc, train_macro_f1, _, _, _, _ = evaluate(
                model=model,
                loader=train_eval_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
            )

            val_loss, val_acc, val_macro_f1, val_bal_acc, y_val_true_ep, y_val_pred_ep, y_val_proba_ep = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
            )
        else:
            train_loss = math.nan
            train_acc = math.nan
            train_macro_f1 = math.nan
            val_loss = math.nan
            val_acc = math.nan
            val_macro_f1 = math.nan
            val_bal_acc = math.nan
            y_val_true_ep = np.zeros((0,), dtype=np.int64)
            y_val_pred_ep = np.zeros((0,), dtype=np.int64)
            y_val_proba_ep = np.zeros((0, num_classes), dtype=np.float32)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.time() - epoch_start

        history.append(
            EpochMetrics(
                epoch=epoch,
                lr=current_lr,
                train_loss=train_loss,
                train_acc=train_acc,
                train_macro_f1=train_macro_f1,
                val_loss=val_loss,
                val_acc=val_acc,
                val_macro_f1=val_macro_f1,
                val_balanced_acc=val_bal_acc,
                epoch_seconds=epoch_seconds,
            )
        )

        if should_eval:
            log(
                f"Epoch {epoch:03d} | "
                f"lr={current_lr:.6g} | "
                f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} train_macro_f1={train_macro_f1:.4f} | "
                f"val_loss={val_loss:.5f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} val_bal_acc={val_bal_acc:.4f} | "
                f"time={epoch_seconds:.1f}s"
            )
        else:
            log(
                f"Epoch {epoch:03d} | "
                f"lr={current_lr:.6g} | "
                f"time={epoch_seconds:.1f}s"
            )

        if scheduler is not None and should_eval and np.isfinite(val_loss):
            scheduler.step(val_loss)

        improved = False
        if should_eval and np.isfinite(val_macro_f1):
            if val_macro_f1 > best_val_macro_f1 + args.min_delta:
                improved = True
            elif abs(val_macro_f1 - best_val_macro_f1) <= args.min_delta and val_loss < best_val_loss:
                improved = True

        if improved:
            best_val_macro_f1 = float(val_macro_f1)
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_val_bal_acc = float(val_bal_acc)
            best_epoch = epoch
            epochs_without_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dims": args.hidden,
                    "num_classes": num_classes,
                    "dropout": args.dropout,
                    "feature_names": feature_names.tolist(),
                    "class_weights": class_weights_np.tolist(),
                    "mu": mu.tolist(),
                    "sigma": sigma.tolist(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "best_val_balanced_acc": best_val_bal_acc,
                    "args": args_serializable,
                    "data_meta": meta,
                },
                best_ckpt_path,
            )

            best_val_cm = confusion_matrix_np(y_val_true_ep, y_val_pred_ep, num_classes)
            save_confusion_matrix_csv(cm_val_csv_path, best_val_cm)
            save_predictions_csv(val_preds_csv_path, y_val_true_ep, y_val_pred_ep, y_val_proba_ep)
            save_per_class_metrics_csv(
                per_class_val_csv_path,
                per_class_metrics_from_cm(best_val_cm),
            )

            log(
                f"New best model saved at epoch {epoch} "
                f"with val_macro_f1={best_val_macro_f1:.4f}, val_loss={best_val_loss:.5f}"
            )
        elif should_eval:
            epochs_without_improve += 1
            log(
                f"No improvement for {epochs_without_improve} eval(s). "
                f"Best epoch={best_epoch}, best_val_macro_f1={best_val_macro_f1:.4f}, best_val_loss={best_val_loss:.5f}"
            )

        save_history_csv(history_csv_path, history)

        if should_eval and epochs_without_improve >= args.patience:
            log("Early stopping triggered.")
            break

    total_seconds = time.time() - overall_start

    test_loss = math.nan
    test_acc = math.nan
    test_macro_f1 = math.nan
    test_bal_acc = math.nan
    test_done = False

    if best_ckpt_path.exists() and test_loader is not None:
        log("Loading best checkpoint for final test evaluation.")
        best_model, _ = load_best_checkpoint_for_eval(best_ckpt_path, device=device)

        test_loss, test_acc, test_macro_f1, test_bal_acc, y_test_true_best, y_test_pred_best, y_test_proba_best = evaluate(
            model=best_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        test_cm = confusion_matrix_np(y_test_true_best, y_test_pred_best, num_classes)
        save_confusion_matrix_csv(cm_test_csv_path, test_cm)
        save_predictions_csv(test_preds_csv_path, y_test_true_best, y_test_pred_best, y_test_proba_best)
        save_per_class_metrics_csv(
            per_class_test_csv_path,
            per_class_metrics_from_cm(test_cm),
        )

        log(
            f"Test metrics | "
            f"loss={test_loss:.5f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f} bal_acc={test_bal_acc:.4f}"
        )
        test_done = True
    elif test_loader is None:
        log("No test split found in NPZ. Skipping test evaluation.")

    summary = {
        "created_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "data": str(args.data),
        "outdir": str(args.outdir),
        "model": "MLP",
        "model_type": "mlp_classifier",
        "seed": args.seed,
        "device": str(device),
        "input_dim": input_dim,
        "expected_input_dim": EXPECTED_INPUT_DIM,
        "num_classes": num_classes,
        "hidden_dims": args.hidden,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "class_weights": class_weights_np.tolist(),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]) if has_test and X_test is not None else 0,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc) if np.isfinite(best_val_acc) else None,
        "best_val_balanced_acc": float(best_val_bal_acc) if np.isfinite(best_val_bal_acc) else None,
        "test_evaluated": test_done,
        "test_loss": float(test_loss) if np.isfinite(test_loss) else None,
        "test_acc": float(test_acc) if np.isfinite(test_acc) else None,
        "test_macro_f1": float(test_macro_f1) if np.isfinite(test_macro_f1) else None,
        "test_balanced_acc": float(test_bal_acc) if np.isfinite(test_bal_acc) else None,
        "total_train_seconds": float(total_seconds),
        "history_csv": str(history_csv_path),
        "best_model_path": str(best_ckpt_path),
        "confusion_matrix_val_csv": str(cm_val_csv_path),
        "val_predictions_csv": str(val_preds_csv_path),
        "per_class_metrics_val_csv": str(per_class_val_csv_path),
        "confusion_matrix_test_csv": str(cm_test_csv_path) if test_done else None,
        "test_predictions_csv": str(test_preds_csv_path) if test_done else None,
        "per_class_metrics_test_csv": str(per_class_test_csv_path) if test_done else None,
        "train_label_presence": train_presence,
        "val_label_presence": val_presence,
        "test_label_presence": test_presence,
        "data_meta": meta,
        "args": args_serializable,
    }

    write_json(summary_json_path, summary)

    log(f"Training finished in {total_seconds:.1f}s")
    log(f"Best epoch            : {best_epoch}")
    log(f"Best val macro F1     : {best_val_macro_f1:.4f}")
    log(f"Best val loss         : {best_val_loss:.5f}")
    if test_done:
        log(f"Test macro F1         : {test_macro_f1:.4f}")
        log(f"Test balanced acc     : {test_bal_acc:.4f}")
    log(f"Saved summary         : {summary_json_path}")
    log(f"Saved history         : {history_csv_path}")
    log(f"Saved best model      : {best_ckpt_path}")
    log(f"Saved val confusion   : {cm_val_csv_path}")
    log(f"Saved val predictions : {val_preds_csv_path}")
    log(f"Saved val per-class   : {per_class_val_csv_path}")
    if test_done:
        log(f"Saved test confusion  : {cm_test_csv_path}")
        log(f"Saved test predictions: {test_preds_csv_path}")
        log(f"Saved test per-class  : {per_class_test_csv_path}")


if __name__ == "__main__":
    main()
