#!/usr/bin/env python3
"""
Train AE64->MLP (v3) with:
- WeightedRandomSampler (class-balanced training)
- ReduceLROnPlateau scheduler on val_macro_f1 (mode=max)
- Logit adjustment: logits_adj = logits + tau * log(prior)
- Focal Loss + label smoothing
- tau sweep: --tau-list "1.2,1.3,1.4,1.5,1.6,1.7"
- multi-seed loop + ensemble evaluation (average probs across seeds on val)

Example single run:
  python scripts/training/train_mlp_ae64_v3.py \
    --data data/processed/training/ae64_samples_4upazila_2023_v3.npz \
    --run-name ae64_v3_tau15_seed42 \
    --epochs 60 --batch-size 4096 \
    --lr 3e-4 --weight-decay 1e-3 --dropout 0.3 \
    --tau 1.5 --amp --seed 42

Example tau sweep + 5 seeds + ensemble:
  python scripts/training/train_mlp_ae64_v3.py \
    --data data/processed/training/ae64_samples_4upazila_2023_v3.npz \
    --run-name ae64_v3_sweep \
    --epochs 60 --batch-size 4096 \
    --lr 3e-4 --weight-decay 1e-3 --dropout 0.3 \
    --tau-list "1.2,1.3,1.4,1.5,1.6,1.7" \
    --seeds "42,43,44,45,46" \
    --amp --ensemble-eval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ----------------------
# Utilities
# ----------------------
def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def human(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000.0:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


# ----------------------
# Metrics
# ----------------------
@torch.no_grad()
def confusion_matrix(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def per_class_metrics(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    support = cm.sum(axis=1).astype(np.int64)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1 = per_class_metrics(cm)["f1"]
    return float(np.mean(f1))


# ----------------------
# Dataset
# ----------------------
class NpzPixelDataset(Dataset):
    def __init__(self, X: np.ndarray, y0: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        self.X = X.astype(np.float32, copy=False)
        self.y = y0.astype(np.int64, copy=False)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (self.X[idx] - self.mean) / self.std
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def make_weighted_sampler(y0: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(y0, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv[y0]
    weights_t = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_t, num_samples=len(y0), replacement=True)


# ----------------------
# Model
# ----------------------
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, use_bn: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class MLPResNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float, use_bn: bool, num_classes: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.bn_in = nn.BatchNorm1d(hidden) if use_bn else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        blocks = []
        for _ in range(depth):
            blocks.append(ResidualMLPBlock(hidden, dropout=dropout, use_bn=use_bn))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.bn_in(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.blocks(h)
        return self.out(h)


# ----------------------
# Losses
# ----------------------
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        with torch.no_grad():
            y_onehot = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1.0)
            if self.label_smoothing > 0:
                y_onehot = y_onehot * (1.0 - self.label_smoothing) + self.label_smoothing / n_classes

        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        p_t = (p * y_onehot).sum(dim=1)
        logp_t = (logp * y_onehot).sum(dim=1)

        focal = (1.0 - p_t).clamp_min(1e-6).pow(self.gamma)
        loss = -focal * logp_t

        if self.alpha is not None:
            alpha_t = (self.alpha.unsqueeze(0) * y_onehot).sum(dim=1)
            loss = loss * alpha_t

        return loss.mean()


def compute_class_prior(y0: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y0, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    prior = counts / counts.sum()
    return prior


# ----------------------
# Training / Evaluation
# ----------------------
@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float,
    log_prior_t: torch.Tensor,
    amp: bool,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_n = 0
    total_nll = 0.0
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            if tau > 0:
                logits = logits + float(tau) * log_prior_t
            logp = F.log_softmax(logits, dim=1)
            nll = F.nll_loss(logp, y, reduction="mean")

        bs = y.size(0)
        total_n += bs
        total_nll += float(nll.item()) * bs

        pred = logits.argmax(dim=1)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

        p = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
        probs_all.append(p)

    y_true = np.concatenate(y_true_all) if y_true_all else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.zeros((0,), dtype=np.int64)
    probs = np.concatenate(probs_all) if probs_all else np.zeros((0, 10), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if total_n > 0 else 0.0
    cm = confusion_matrix(num_classes=log_prior_t.numel(), y_true=y_true, y_pred=y_pred)
    macro_f1 = macro_f1_from_cm(cm)
    avg_nll = total_nll / max(1, total_n)
    return avg_nll, acc, macro_f1, y_true, y_pred, probs


def save_confusion_outputs(
    cm: np.ndarray,
    out_dir: Path,
    class_names_1based: Dict[int, str],
    prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)

    csv_path = out_dir / f"{prefix}_confusion_matrix.csv"
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d")

    txt_path = out_dir / f"{prefix}_confusion_matrix.txt"
    labels = [class_names_1based[i + 1] for i in range(cm.shape[0])]
    with txt_path.open("w") as f:
        f.write("Confusion Matrix (rows=true, cols=pred)\n\n")
        f.write("Labels (0-based idx -> 1-based class -> name):\n")
        for i, name in enumerate(labels):
            f.write(f"  {i:2d} -> {i+1:2d} -> {name}\n")
        f.write("\nMatrix:\n")
        for r in range(cm.shape[0]):
            f.write(" ".join(f"{cm[r, c]:7d}" for c in range(cm.shape[1])) + "\n")

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix (val)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i + 1) for i in range(cm.shape[0])], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([str(i + 1) for i in range(cm.shape[0])], fontsize=9)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close(fig)


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
    label_smoothing: float
    focal_gamma: float
    focal_alpha: str
    tau: float
    amp: bool
    seed: int
    early_stop_patience: int
    plateau_patience: int
    plateau_factor: float
    min_lr: float


def train_one(
    *,
    data_path: Path,
    run_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    depth: int,
    dropout: float,
    use_bn: bool,
    label_smoothing: float,
    focal_gamma: float,
    focal_alpha: str,
    tau: float,
    amp: bool,
    seed: int,
    early_stop_patience: int,
    plateau_patience: int,
    plateau_factor: float,
    min_lr: float,
    writer: SummaryWriter | None,
) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    set_seed(seed)
    dev = torch.device("cuda")

    npz = np.load(data_path, allow_pickle=True)
    X_train = npz["X_train"].astype(np.float32, copy=False)
    y_train = npz["y_train"].astype(np.int64, copy=False)
    X_val = npz["X_val"].astype(np.float32, copy=False)
    y_val = npz["y_val"].astype(np.int64, copy=False)

    y_train0 = y_train - 1
    y_val0 = y_val - 1
    num_classes = 10

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    np.savez_compressed(run_dir / "norm_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    train_ds = NpzPixelDataset(X_train, y_train0, mean, std)
    val_ds = NpzPixelDataset(X_val, y_val0, mean, std)

    sampler = make_weighted_sampler(y_train0, num_classes=num_classes)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    prior = compute_class_prior(y_train0, num_classes=num_classes)
    log_prior = np.log(prior + 1e-12).astype(np.float32)
    log_prior_t = torch.tensor(log_prior, device=dev).view(1, -1)

    alpha_t = None
    if focal_alpha == "balanced":
        counts = np.bincount(y_train0, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        w = 1.0 / counts
        w = w * (num_classes / w.sum())
        alpha_t = torch.tensor(w.astype(np.float32), device=dev)
    elif focal_alpha == "none":
        alpha_t = None
    else:
        raise ValueError("focal_alpha must be 'balanced' or 'none'.")

    criterion = FocalLoss(gamma=focal_gamma, alpha=alpha_t, label_smoothing=label_smoothing)

    model = MLPResNet(
        in_dim=X_train.shape[1],
        hidden=hidden,
        depth=depth,
        dropout=dropout,
        use_bn=use_bn,
        num_classes=num_classes,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=plateau_factor,
        patience=plateau_patience,
        threshold=1e-4,
        min_lr=min_lr,
        verbose=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(amp and dev.type == "cuda"))

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    best_macro_f1 = -1.0
    best_epoch = -1
    best_val_loss_like = 1e9
    patience_left = early_stop_patience

    log(f"Run: {run_dir}")
    log(f"Train: X={X_train.shape} y={y_train0.shape} | Val: X={X_val.shape} y={y_val0.shape}")
    log(f"Model params: {human(sum(p.numel() for p in model.parameters()))}")
    log(f"Loss: focal | label_smoothing={label_smoothing} | focal_gamma={focal_gamma} | focal_alpha={focal_alpha}")
    log(f"Logit adjustment tau={tau} (0 disables)")
    log(f"Sampler: WeightedRandomSampler (class-balanced)")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=dev.type, enabled=(amp and dev.type == "cuda")):
                logits = model(x)
                if tau > 0:
                    logits = logits + float(tau) * log_prior_t
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()

            bs = y.size(0)
            seen += bs
            running_loss += float(loss.item()) * bs
            running_acc += float(acc) * bs

            if step == 1 or step % 50 == 0 or step == len(train_loader):
                lr_now = optimizer.param_groups[0]["lr"]
                log(
                    f"Epoch {epoch:03d}/{epochs} Step {step:05d}/{len(train_loader)} "
                    f"lr={lr_now:.2e} loss={running_loss/max(1,seen):.4f} acc={running_acc/max(1,seen):.4f}"
                )

        train_loss = running_loss / max(1, seen)
        train_acc = running_acc / max(1, seen)

        val_loss_like, val_acc, val_macro_f1, y_true, y_pred, _ = eval_epoch(
            model=model,
            loader=val_loader,
            device=dev,
            tau=tau,
            log_prior_t=log_prior_t,
            amp=(amp and dev.type == "cuda"),
        )

        scheduler.step(val_macro_f1)

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        cm = confusion_matrix(num_classes=num_classes, y_true=y_true, y_pred=y_pred)
        m = per_class_metrics(cm)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss_like": float(val_loss_like),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "lr": float(lr_now),
            "sec": float(dt),
            "per_class_f1": m["f1"].tolist(),
            "per_class_precision": m["precision"].tolist(),
            "per_class_recall": m["recall"].tolist(),
            "support": m["support"].tolist(),
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("val/loss_like", val_loss_like, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
            writer.add_scalar("val/macro_f1", val_macro_f1, epoch)
            writer.add_scalar("lr", lr_now, epoch)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "tau": float(tau),
                "seed": int(seed),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "dropout": float(dropout),
                "hidden": int(hidden),
                "depth": int(depth),
                "use_bn": bool(use_bn),
            },
            run_dir / "last.pt",
        )

        improved = val_macro_f1 > best_macro_f1 + 1e-6
        if improved:
            best_macro_f1 = float(val_macro_f1)
            best_val_loss_like = float(val_loss_like)
            best_epoch = epoch
            patience_left = early_stop_patience
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                    "tau": float(tau),
                    "seed": int(seed),
                },
                run_dir / "best.pt",
            )

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
            save_confusion_outputs(cm, run_dir, class_names, prefix="best_val")
        else:
            patience_left -= 1

        log(
            f"Epoch {epoch:03d} done in {dt:.1f}s | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss_like={val_loss_like:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} | "
            f"best_macro_f1={best_macro_f1:.4f} (epoch {best_epoch}) | patience_left={patience_left}"
        )

        if patience_left <= 0:
            log("Early stopping (macro-F1) triggered.")
            break

    summary = {
        "best_macro_f1": float(best_macro_f1),
        "best_epoch": int(best_epoch),
        "best_val_loss_like": float(best_val_loss_like),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Finished. Best macro-F1: {best_macro_f1:.4f} at epoch {best_epoch} (val_loss_like={best_val_loss_like:.4f})")
    return summary


@torch.no_grad()
def ensemble_eval(
    data_path: Path,
    run_dirs: List[Path],
    tau: float,
    out_path: Path,
    batch_size: int,
    amp: bool,
) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    dev = torch.device("cuda")

    npz = np.load(data_path, allow_pickle=True)
    X_val = npz["X_val"].astype(np.float32, copy=False)
    y_val = (npz["y_val"].astype(np.int64, copy=False) - 1)

    first = torch.load(run_dirs[0] / "best.pt", map_location="cpu")
    mean = first["mean"]
    std = first["std"]
    std = np.where(std < 1e-6, 1.0, std)

    ds = NpzPixelDataset(X_val, y_val, mean, std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    X_train = npz["X_train"].astype(np.float32, copy=False)
    y_train0 = (npz["y_train"].astype(np.int64, copy=False) - 1)
    prior = compute_class_prior(y_train0, num_classes=10)
    log_prior_t = torch.tensor(np.log(prior + 1e-12).astype(np.float32), device=dev).view(1, -1)

    models: List[nn.Module] = []
    for rd in run_dirs:
        ckpt = torch.load(rd / "best.pt", map_location="cpu")
        model = MLPResNet(in_dim=64, hidden=512, depth=3, dropout=0.3, use_bn=True, num_classes=10).to(dev)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        models.append(model)

    probs_sum = None
    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)

        with torch.autocast(device_type=dev.type, enabled=(amp and dev.type == "cuda")):
            probs_batch = None
            for model in models:
                logits = model(x)
                if tau > 0:
                    logits = logits + float(tau) * log_prior_t
                p = F.softmax(logits, dim=1)
                probs_batch = p if probs_batch is None else (probs_batch + p)
            probs_batch = probs_batch / float(len(models))

        pred = probs_batch.argmax(dim=1)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

        if probs_sum is None:
            probs_sum = probs_batch.detach().cpu().numpy().astype(np.float32)
        else:
            probs_sum = np.concatenate([probs_sum, probs_batch.detach().cpu().numpy().astype(np.float32)], axis=0)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    cm = confusion_matrix(10, y_true, y_pred)
    macro_f1 = macro_f1_from_cm(cm)
    acc = float((y_true == y_pred).mean())

    result = {"ensemble_val_macro_f1": float(macro_f1), "ensemble_val_acc": float(acc)}
    out_path.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AE64 MLP (v3) with sampler + plateau scheduler + tau sweep + ensemble.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs"))

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-bn", action="store_true")

    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=str, default="balanced", choices=["balanced", "none"])

    p.add_argument("--tau", type=float, default=1.5)
    p.add_argument("--tau-list", type=str, default="", help="Comma-separated tau list, e.g. '1.2,1.3,...'")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, e.g. '42,43,44,45,46'")
    p.add_argument("--ensemble-eval", action="store_true")

    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=3e-6)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--tensorboard", action="store_true")
    return p.parse_args()


def parse_list_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_ints(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    taus = parse_list_floats(args.tau_list) if args.tau_list else [float(args.tau)]
    seeds = parse_list_ints(args.seeds) if args.seeds else [int(args.seed)]

    base_dir = args.out_dir / args.run_name
    ensure_dir(base_dir)

    high_cfg = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden": args.hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "use_bn": (not args.no_bn),
        "label_smoothing": args.label_smoothing,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "taus": taus,
        "seeds": seeds,
        "early_stop_patience": args.early_stop_patience,
        "plateau_patience": args.plateau_patience,
        "plateau_factor": args.plateau_factor,
        "min_lr": args.min_lr,
        "amp": args.amp,
    }
    (base_dir / "sweep_config.json").write_text(json.dumps(high_cfg, indent=2))

    all_results = []

    for tau in taus:
        for seed in seeds:
            run_dir = base_dir / f"tau{tau:.2f}_seed{seed}"
            ensure_dir(run_dir)
            (run_dir / "config.json").write_text(json.dumps(high_cfg | {"tau": tau, "seed": seed}, indent=2))

            writer = None
            if args.tensorboard and SummaryWriter is not None:
                ensure_dir(run_dir / "tb")
                writer = SummaryWriter(log_dir=str(run_dir / "tb"))

            summary = train_one(
                data_path=args.data,
                run_dir=run_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden=args.hidden,
                depth=args.depth,
                dropout=args.dropout,
                use_bn=(not args.no_bn),
                label_smoothing=args.label_smoothing,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                tau=tau,
                amp=args.amp,
                seed=seed,
                early_stop_patience=args.early_stop_patience,
                plateau_patience=args.plateau_patience,
                plateau_factor=args.plateau_factor,
                min_lr=args.min_lr,
                writer=writer,
            )

            if writer is not None:
                writer.flush()
                writer.close()

            all_results.append({"tau": tau, "seed": seed} | summary)

    (base_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))

    if args.ensemble_eval and len(seeds) >= 2:
        for tau in taus:
            run_dirs = [base_dir / f"tau{tau:.2f}_seed{s}" for s in seeds]
            ok = all((rd / "best.pt").exists() for rd in run_dirs)
            if not ok:
                log(f"[WARN] Skipping ensemble for tau={tau:.2f}: missing best.pt in some seeds.")
                continue
            out_path = base_dir / f"ensemble_tau{tau:.2f}.json"
            res = ensemble_eval(
                data_path=args.data,
                run_dirs=run_dirs,
                tau=tau,
                out_path=out_path,
                batch_size=args.batch_size,
                amp=args.amp,
            )
            log(f"Ensemble tau={tau:.2f}: macro_f1={res['ensemble_val_macro_f1']:.4f} acc={res['ensemble_val_acc']:.4f}")

    log(f"Done. Outputs under: {base_dir}")


if __name__ == "__main__":
    main()
