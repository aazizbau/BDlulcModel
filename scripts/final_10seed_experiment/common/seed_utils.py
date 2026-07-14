"""Seed helpers for repeated-seed experiments."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - only used when torch is unavailable.
    torch = None


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch random seeds when PyTorch is available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as exc:
            print(f"Warning: complete deterministic execution could not be enabled: {exc}")


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python random inside a PyTorch DataLoader worker."""
    if torch is None:
        worker_seed = worker_id
    else:
        worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

