#!/usr/bin/env python3
"""
Improved MLP training for AE64 embeddings.

Adds:
  - Macro-F1 + per-class metrics each epoch; best checkpoint by macro-F1
  - Optional logit adjustment: logits_adj = logits + tau * log(prior)
  - Default label smoothing = 0.05 (for CE); focal loss option
  - Residual MLP blocks: Linear -> BN -> GELU -> Dropout -> Linear + skip
  - --no-bn to disable BN
  - Optional grid search over lr/wd/dropout
  - Confusion matrix: PNG with values + CSV + TXT

NPZ input keys expected:
  X_train (N,D) float32
  y_train (N,)  uint8/int  classes 1..K
  X_val   (M,D) float32
  y_val   (M,)  uint8/int  classes 1..K
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ------------------------
# utils
# ------------------------
def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def human(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000.0:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


# ------------------------
# dataset
# ------------------------
class NpzPixelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: str = "zscore") -> None:
        assert X.ndim == 2 and y.ndim == 1
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.normalize = normalize
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def set_norm_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.normalize == "zscore":
            if self.mean is None or self.std is None:
                raise RuntimeError("Normalization stats not set.")
            x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


# ------------------------
# metrics
# ------------------------
@torch.no_grad()
def confusion_matrix(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def classification_report_from_cm(cm: np.ndarray, eps: float = 1e-12) -> Dict[str, object]:
    C = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    true_sum = cm.sum(axis=1).astype(np.float64)

    precision = tp / np.maximum(pred_sum, eps)
    recall = tp / np.maximum(true_sum, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)

    macro_p = float(np.mean(precision))
    macro_r = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    per_class = []
    for i in range(C):
        per_class.append(
            {
                "class": int(i),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(true_sum[i]),
            }
        )

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def save_confusion_csv(cm: np.ndarray, out_csv: Path, class_names: Optional[Dict[int, str]] = None) -> None:
    C = cm.shape[0]
    labels = []
    for i in range(C):
        key = i + 1
        labels.append(class_names[key] if (class_names and key in class_names) else str(key))

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i in range(C):
            w.writerow([labels[i]] + cm[i, :].tolist())


def save_confusion_txt(cm: np.ndarray, out_txt: Path, class_names: Optional[Dict[int, str]] = None) -> None:
    C = cm.shape[0]
    labels = []
    for i in range(C):
        key = i + 1
        labels.append(class_names[key] if (class_names and key in class_names) else str(key))

    lines = []
    lines.append("Confusion Matrix (rows=true, cols=pred)\n")
    header = "true\\pred".ljust(24) + " ".join([lab[:18].ljust(18) for lab in labels])
    lines.append(header)
    for i in range(C):
        row = labels[i][:22].ljust(24) + " ".join([str(int(v)).ljust(18) for v in cm[i, :]])
        lines.append(row)

    out_txt.write_text("\n".join(lines))


def save_confusion_png_with_values(
    cm: np.ndarray,
    out_png: Path,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    import matplotlib.pyplot as plt

    C = cm.shape[0]
    labels = []
    for i in range(C):
        key = i + 1
        labels.append(class_names[key] if (class_names and key in class_names) else str(key))

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111)

    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix (val)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ticks = np.arange(C)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(C):
        for j in range(C):
            v = int(cm[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_curves_png(history: Dict[str, list], out_path_prefix: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig1 = plt.figure(figsize=(8, 5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(epochs, history["train_loss"], label="train_loss")
    ax1.plot(epochs, history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_path_prefix.with_name(out_path_prefix.stem + "_loss.png"), dpi=160)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(epochs, history["train_acc"], label="train_acc")
    ax2.plot(epochs, history["val_acc"], label="val_acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_path_prefix.with_name(out_path_prefix.stem + "_acc.png"), dpi=160)
    plt.close(fig2)

    if "val_macro_f1" in history:
        fig3 = plt.figure(figsize=(8, 5))
        ax3 = fig3.add_subplot(111)
        ax3.plot(epochs, history["val_macro_f1"], label="val_macro_f1")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Macro-F1")
        ax3.set_title("Macro-F1 Curve")
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(out_path_prefix.with_name(out_path_prefix.stem + "_macrof1.png"), dpi=160)
        plt.close(fig3)


# ------------------------
# model
# ------------------------
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, use_bn: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class ResidualMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden: int = 512,
        depth: int = 3,
        dropout: float = 0.2,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden, dropout=dropout, use_bn=use_bn) for _ in range(depth)]
        )
        self.out_norm = nn.BatchNorm1d(hidden) if use_bn else nn.Identity()
        self.out_act = nn.GELU()
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.blocks(h)
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.head(h)


# ------------------------
# losses
# ------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        idx = target.unsqueeze(1)
        pt = probs.gather(1, idx).squeeze(1)
        logpt = log_probs.gather(1, idx).squeeze(1)

        loss = -((1.0 - pt) ** self.gamma) * logpt

        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = loss * at

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_priors(y_train_0based: np.ndarray, num_classes: int, eps: float = 1e-12) -> np.ndarray:
    counts = np.bincount(y_train_0based, minlength=num_classes).astype(np.float64)
    priors = counts / max(counts.sum(), 1.0)
    priors = np.clip(priors, eps, 1.0)
    priors = priors / priors.sum()
    return priors


def compute_balanced_alpha_from_priors(priors: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.maximum(priors, 1e-12)
    inv = inv * (len(inv) / inv.sum())
    return inv


# ------------------------
# eval epoch
# ------------------------
@torch.no_grad()
def epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp: bool,
    log_prior_t: Optional[torch.Tensor],
    tau: float,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            if (log_prior_t is not None) and (tau > 0):
                logits = logits + tau * log_prior_t
            loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(dim=1)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.zeros((0,), dtype=np.int64)
    if log_prior_t is not None:
        num_classes = int(log_prior_t.numel())
    else:
        num_classes = int(logits.shape[1])
    cm = confusion_matrix(num_classes=num_classes, y_true=y_true, y_pred=y_pred)

    avg_loss = total_loss / max(1, total_n)
    acc = float((y_true == y_pred).mean()) if total_n > 0 else 0.0
    rep = classification_report_from_cm(cm)
    macro_f1 = float(rep["macro_f1"])
    return avg_loss, acc, macro_f1, y_true, y_pred, cm


# ------------------------
# config
# ------------------------
@dataclass
class TrainConfig:
    data: str
    run_name: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden: int
    depth: int
    dropout: float
    use_bn: bool
    amp: bool
    seed: int
    device: str
    early_stop_patience: int
    scheduler: str
    warmup_epochs: int
    label_smoothing: float
    loss: str
    focal_gamma: float
    focal_alpha: str
    tau: float


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            t = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


# ------------------------
# train one run
# ------------------------
def train_one_run(args: argparse.Namespace, run_dir: Path, lr: float, wd: float, dropout: float) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required. No GPU detected.")
    dev = torch.device("cuda")

    set_seed(args.seed)
    ensure_dir(run_dir)
    ensure_dir(run_dir / "tb")

    npz = np.load(args.data, allow_pickle=True)
    X_train = npz["X_train"].astype(np.float32, copy=False)
    y_train = npz["y_train"].astype(np.int64, copy=False)
    X_val = npz["X_val"].astype(np.float32, copy=False)
    y_val = npz["y_val"].astype(np.int64, copy=False)

    y_train0 = y_train - 1
    y_val0 = y_val - 1
    num_classes = int(args.num_classes)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    train_ds = NpzPixelDataset(X_train, y_train0, normalize="zscore")
    val_ds = NpzPixelDataset(X_val, y_val0, normalize="zscore")
    train_ds.set_norm_stats(mean, std)
    val_ds.set_norm_stats(mean, std)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    priors = compute_priors(y_train0, num_classes=num_classes)
    log_prior_t = torch.tensor(np.log(priors).astype(np.float32), device=dev)

    model = ResidualMLP(
        in_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden=args.hidden,
        depth=args.depth,
        dropout=dropout,
        use_bn=not args.no_bn,
    ).to(dev)

    if args.loss == "ce":
        criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == "focal":
        alpha_t: Optional[torch.Tensor] = None
        if args.focal_alpha == "balanced":
            alpha = compute_balanced_alpha_from_priors(priors)
            alpha_t = torch.tensor(alpha.astype(np.float32), device=dev)
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha_t, reduction="mean")
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    optimizer = make_optimizer(model, lr=lr, weight_decay=wd)
    scheduler = make_scheduler(optimizer, args.scheduler, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and dev.type == "cuda"))
    writer = SummaryWriter(log_dir=str(run_dir / "tb")) if SummaryWriter is not None else None

    cfg = TrainConfig(
        data=str(args.data),
        run_name=str(run_dir.name),
        out_dir=str(run_dir.parent),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(lr),
        weight_decay=float(wd),
        hidden=int(args.hidden),
        depth=int(args.depth),
        dropout=float(dropout),
        use_bn=not args.no_bn,
        amp=bool(args.amp),
        seed=int(args.seed),
        device=str(dev),
        early_stop_patience=int(args.early_stop_patience),
        scheduler=str(args.scheduler),
        warmup_epochs=int(args.warmup_epochs),
        label_smoothing=float(args.label_smoothing),
        loss=str(args.loss),
        focal_gamma=float(args.focal_gamma),
        focal_alpha=str(args.focal_alpha),
        tau=float(args.tau),
    )

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    np.savez_compressed(run_dir / "norm_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    (run_dir / "priors.json").write_text(json.dumps({"priors": priors.tolist()}, indent=2))

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": [], "lr": []}

    best_macro_f1 = -1.0
    best_val_loss_at_best = float("inf")
    best_epoch = -1
    patience_left = int(args.early_stop_patience)

    log(f"Run: {run_dir}")
    log(f"Train: X={X_train.shape} y={y_train0.shape} | Val: X={X_val.shape} y={y_val0.shape}")
    log(f"Model params: {human(sum(p.numel() for p in model.parameters()))}")
    log(f"Loss: {args.loss} | label_smoothing={args.label_smoothing} | focal_gamma={args.focal_gamma} | focal_alpha={args.focal_alpha}")
    log(f"Logit adjustment tau={args.tau} (0 disables)")

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=dev.type, enabled=(args.amp and dev.type == "cuda")):
                logits = model(x)
                if args.tau > 0:
                    logits = logits + args.tau * log_prior_t
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            seen += bs
            running_loss += float(loss.item()) * bs
            running_acc += accuracy_top1(logits.detach(), y) * bs

            if step == 1 or step % 50 == 0 or step == len(train_loader):
                lr_now = optimizer.param_groups[0]["lr"]
                log(
                    f"Epoch {epoch:03d}/{int(args.epochs)} Step {step:05d}/{len(train_loader)} "
                    f"lr={lr_now:.2e} loss={(running_loss/max(1,seen)):.4f} acc={(running_acc/max(1,seen)):.4f}"
                )

        train_loss = running_loss / max(1, seen)
        train_acc = running_acc / max(1, seen)

        val_loss, val_acc, val_macro_f1, y_true, y_pred, cm = epoch_eval(
            model=model,
            loader=val_loader,
            device=dev,
            criterion=criterion,
            amp=(args.amp and dev.type == "cuda"),
            log_prior_t=log_prior_t,
            tau=float(args.tau),
        )

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_macro_f1))
        history["lr"].append(float(lr_now))

        rep = classification_report_from_cm(cm)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_macro_precision": float(rep["macro_precision"]),
            "val_macro_recall": float(rep["macro_recall"]),
            "lr": float(lr_now),
            "sec": float(dt),
            "per_class": rep["per_class"],
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("f1_macro/val", val_macro_f1, epoch)
            writer.add_scalar("lr", lr_now, epoch)
            for pc in rep["per_class"]:
                writer.add_scalar(f"f1/class_{pc['class']}", pc["f1"], epoch)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(cfg),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "priors": priors.astype(np.float32),
            },
            run_dir / "last.pt",
        )

        improved = (val_macro_f1 > best_macro_f1 + 1e-12) or (
            abs(val_macro_f1 - best_macro_f1) <= 1e-12 and val_loss < best_val_loss_at_best - 1e-12
        )

        if improved:
            best_macro_f1 = float(val_macro_f1)
            best_val_loss_at_best = float(val_loss)
            best_epoch = epoch
            patience_left = int(args.early_stop_patience)

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                    "priors": priors.astype(np.float32),
                    "val_macro_f1": best_macro_f1,
                    "val_loss": best_val_loss_at_best,
                },
                run_dir / "best.pt",
            )
            np.save(run_dir / "best_confusion_matrix.npy", cm)
        else:
            patience_left -= 1

        log(
            f"Epoch {epoch:03d} done in {dt:.1f}s | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} | "
            f"best_macro_f1={best_macro_f1:.4f} (epoch {best_epoch}) | patience_left={patience_left}"
        )

        if patience_left <= 0:
            log("Early stopping (macro-F1) triggered.")
            break

    save_curves_png(history, run_dir / "curves.png")

    class_names = {
        1: "Urban / Institutional Built-up",
        2: "Rural Settlement (Homestead Vegetation)",
        3: "Transport & Coastal Embankments",
        4: "Cropland (All Crop Intensities)",
        5: "Tree-based Agroforestry & Orchard",
        6: "Aquaculture & Inland Ponds",
        7: "Canals & Drainage Network",
        8: "Rivers & Estuarine Channels",
        9: "Mangrove Forest",
        10: "Bare / Exposed Coastal Land",
    }

    best_cm_path = run_dir / "best_confusion_matrix.npy"
    if best_cm_path.exists():
        cm_best = np.load(best_cm_path)
    else:
        cm_best = cm

    save_confusion_png_with_values(cm_best, run_dir / "confusion_matrix_best.png", class_names=class_names)
    save_confusion_csv(cm_best, run_dir / "confusion_matrix_best.csv", class_names=class_names)
    save_confusion_txt(cm_best, run_dir / "confusion_matrix_best.txt", class_names=class_names)

    if writer is not None:
        writer.flush()
        writer.close()

    result = {
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_macro_f1),
        "best_val_loss_at_best": float(best_val_loss_at_best),
        "final_val_acc": float(history["val_acc"][-1]) if history["val_acc"] else 0.0,
    }
    (run_dir / "result_summary.json").write_text(json.dumps(result, indent=2))
    log(f"Finished. Outputs in: {run_dir}")
    log(f"Best macro-F1: {best_macro_f1:.4f} at epoch {best_epoch} (val_loss={best_val_loss_at_best:.4f})")
    return result


# ------------------------
# main + grid search
# ------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train improved residual MLP on AE embeddings.")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run-name", type=str, default="ae64_mlp_improved")
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-classes", type=int, default=10)

    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--no-bn", action="store_true", help="Disable BatchNorm (use Identity).")

    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=str, default="balanced", choices=["none", "balanced"])
    ap.add_argument("--tau", type=float, default=1.0, help="Logit adjustment strength; 0 disables. Try 0.5,1.0,1.5")

    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    ap.add_argument("--warmup-epochs", type=int, default=2)

    ap.add_argument("--early-stop-patience", type=int, default=8)

    ap.add_argument("--grid-search", action="store_true", help="Run grid search over lr/wd/dropout.")
    ap.add_argument("--grid-lr", type=str, default="1e-3,7e-4,3e-4")
    ap.add_argument("--grid-wd", type=str, default="1e-4,5e-4,1e-3,5e-3")
    ap.add_argument("--grid-dropout", type=str, default="0.1,0.2,0.3")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required. No GPU detected.")

    ensure_dir(args.out_dir)

    def parse_float_list(s: str) -> List[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    if args.grid_search:
        lrs = parse_float_list(args.grid_lr)
        wds = parse_float_list(args.grid_wd)
        drops = parse_float_list(args.grid_dropout)

        grid_dir = args.out_dir / f"{args.run_name}_grid"
        ensure_dir(grid_dir)

        results_csv = grid_dir / "grid_results.csv"
        with results_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lr", "weight_decay", "dropout", "best_epoch", "best_macro_f1", "best_val_loss_at_best", "run_dir"])

        all_results = []
        idx = 0
        total = len(lrs) * len(wds) * len(drops)
        log(f"Grid search: {total} runs (lr={lrs}, wd={wds}, dropout={drops})")

        for lr, wd, dr in product(lrs, wds, drops):
            idx += 1
            run_dir = grid_dir / f"run_{idx:03d}_lr{lr:g}_wd{wd:g}_dr{dr:g}"
            log(f"[{idx}/{total}] lr={lr} wd={wd} dropout={dr}")
            res = train_one_run(args, run_dir=run_dir, lr=lr, wd=wd, dropout=dr)
            all_results.append({"lr": lr, "weight_decay": wd, "dropout": dr, **res})

            with results_csv.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow([lr, wd, dr, res["best_epoch"], res["best_macro_f1"], res["best_val_loss_at_best"], res["run_dir"]])

        best = sorted(all_results, key=lambda r: (-r["best_macro_f1"], r["best_val_loss_at_best"]))[0]
        (grid_dir / "grid_best.json").write_text(json.dumps(best, indent=2))
        log(f"Grid best: macro-F1={best['best_macro_f1']:.4f} val_loss={best['best_val_loss_at_best']:.4f} in {best['run_dir']}")
        return

    run_dir = args.out_dir / args.run_name
    train_one_run(args, run_dir=run_dir, lr=float(args.lr), wd=float(args.weight_decay), dropout=float(args.dropout))


if __name__ == "__main__":
    main()
