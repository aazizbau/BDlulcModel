#!/usr/bin/env python3
"""
Train an MLP classifier (pixel-wise) on AlphaEarth 64D embeddings using PyTorch.

Input:
  NPZ from your extractor containing:
    X_train (N,64) float32
    y_train (N,)   uint8  (classes 1..10)
    X_val   (M,64) float32
    y_val   (M,)   uint8

Outputs:
  - runs/<run_name>/ (logs, metrics.jsonl, curves_*.png, best.pt, last.pt, confusion_matrix.png)
  - Optional TensorBoard logs (runs/<run_name>/tb)

Example:
  python scripts/training/train_mlp_ae64.py \
    --data data/processed/training/ae64_samples_4upazila_2023.npz \
    --run-name ae64_mlp_v1 \
    --epochs 40 --batch-size 4096 --lr 3e-4 --weight-decay 1e-4 \
    --amp

Then:
  tensorboard --logdir runs/ae64_mlp_v1/tb
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
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


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


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class NpzPixelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: str = "zscore") -> None:
        assert X.ndim == 2
        assert y.ndim == 1
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

        self.normalize = normalize
        self.mean = None
        self.std = None

    def set_norm_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.normalize == "zscore":
            x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        num_classes: int = 10,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.2,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def confusion_matrix(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_confusion_png(cm: np.ndarray, out_path: Path, class_names: Dict[int, str] | None = None) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix (val)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    n = cm.shape[0]
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if class_names:
        labels = [class_names[i + 1] if (i + 1) in class_names else str(i + 1) for i in range(n)]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_xticklabels([str(i + 1) for i in range(n)], rotation=45, ha="right")
        ax.set_yticklabels([str(i + 1) for i in range(n)])

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_curves_png(history: Dict[str, list], out_path: Path) -> None:
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
    fig1.savefig(out_path.with_name(out_path.stem + "_loss.png"), dpi=160)
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
    fig2.savefig(out_path.with_name(out_path.stem + "_acc.png"), dpi=160)
    plt.close(fig2)


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
    class_weights: str
    grad_clip: float
    num_workers: int
    amp: bool
    seed: int
    device: str
    early_stop_patience: int
    scheduler: str
    warmup_epochs: int


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
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


def compute_class_weights(y_train_0based: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train_0based, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp: bool,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        pred = logits.argmax(dim=1)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.zeros((0,), dtype=np.int64)

    avg_loss = total_loss / max(1, total_n)
    acc = float((y_true == y_pred).mean()) if total_n > 0 else 0.0
    return avg_loss, acc, y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PyTorch MLP on AE64 pixel samples.")
    ap.add_argument("--data", type=Path, required=True, help="NPZ file from extractor.")
    ap.add_argument("--run-name", type=str, default="ae64_mlp", help="Run name (folder under runs/).")
    ap.add_argument("--out-dir", type=Path, default=Path("runs"), help="Base output directory.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no-bn", action="store_true", help="Disable BatchNorm.")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="CrossEntropy label smoothing.")
    ap.add_argument(
        "--class-weights",
        type=str,
        default="balanced",
        choices=["none", "balanced"],
        help="Use class weights in loss.",
    )
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm (0 to disable).")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (GPU recommended).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    ap.add_argument("--early-stop-patience", type=int, default=8, help="Stop if val loss not improving.")
    ap.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
        help="LR scheduler.",
    )
    ap.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs for cosine scheduler.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required. No GPU detected.")
    dev = torch.device("cuda")

    set_seed(args.seed)

    run_dir = args.out_dir / args.run_name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "tb")

    npz = np.load(args.data, allow_pickle=True)
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

    train_ds = NpzPixelDataset(X_train, y_train0, normalize="zscore")
    val_ds = NpzPixelDataset(X_val, y_val0, normalize="zscore")
    train_ds.set_norm_stats(mean, std)
    val_ds.set_norm_stats(mean, std)

    pin = (dev.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = MLP(
        in_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        use_bn=not args.no_bn,
    ).to(dev)

    weight_t = None
    if args.class_weights == "balanced":
        weight_t = compute_class_weights(y_train0, num_classes).to(dev)

    criterion = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=args.label_smoothing)

    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.scheduler, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and dev.type == "cuda"))

    cfg = TrainConfig(
        data=str(args.data),
        run_name=args.run_name,
        out_dir=str(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        use_bn=not args.no_bn,
        label_smoothing=args.label_smoothing,
        class_weights=args.class_weights,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        amp=args.amp,
        seed=args.seed,
        device=str(dev),
        early_stop_patience=args.early_stop_patience,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
    )

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    np.savez_compressed(run_dir / "norm_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    writer = SummaryWriter(log_dir=str(run_dir / "tb")) if SummaryWriter is not None else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = args.early_stop_patience

    log(f"Device: {dev}")
    log(f"Train: X={X_train.shape} y={y_train0.shape}  (classes 0..9)")
    log(f"Val  : X={X_val.shape} y={y_val0.shape}")
    log(f"Batch size: {args.batch_size} | Steps/epoch: {len(train_loader)}")
    log(f"Model params: {human(sum(p.numel() for p in model.parameters()))}")

    for epoch in range(1, args.epochs + 1):
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
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            seen += bs
            running_loss += loss.item() * bs
            running_acc += accuracy_top1(logits.detach(), y) * bs

            if step == 1 or step % 50 == 0 or step == len(train_loader):
                lr_now = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / max(1, seen)
                avg_acc = running_acc / max(1, seen)
                log(
                    f"Epoch {epoch:03d}/{args.epochs} "
                    f"Step {step:05d}/{len(train_loader)} "
                    f"lr={lr_now:.2e} loss={avg_loss:.4f} acc={avg_acc:.4f}"
                )

        train_loss = running_loss / max(1, seen)
        train_acc = running_acc / max(1, seen)

        val_loss, val_acc, y_true, y_pred = epoch_eval(
            model=model,
            loader=val_loader,
            device=dev,
            criterion=criterion,
            amp=(args.amp and dev.type == "cuda"),
        )

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(lr_now))

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(lr_now),
            "sec": float(dt),
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("lr", lr_now, epoch)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(cfg),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            },
            run_dir / "last.pt",
        )

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_left = args.early_stop_patience
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                },
                run_dir / "best.pt",
            )
        else:
            patience_left -= 1

        log(
            f"Epoch {epoch:03d} done in {dt:.1f}s | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"best_val_loss={best_val_loss:.4f} (epoch {best_epoch}) | "
            f"patience_left={patience_left}"
        )

        if patience_left <= 0:
            log("Early stopping triggered.")
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

    if y_true.size > 0:
        cm = confusion_matrix(num_classes, y_true, y_pred)
        np.save(run_dir / "confusion_matrix.npy", cm)
        save_confusion_png(cm, run_dir / "confusion_matrix.png", class_names=class_names)

    if writer is not None:
        writer.flush()
        writer.close()

    log(f"Finished. Outputs in: {run_dir}")
    log(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
