"""Seed helpers for repeated-seed experiments.

Reproduction and AOI adaptation
-------------------------------
Workflow role: Run or summarize repeated-seed experiments for empirical model-performance uncertainty.

Run commands from the repository root after activating the project environment and
installing ``requirements.txt``. Keep immutable raw inputs separate from generated
intermediate and output products, and create a new output directory for each AOI/run.

Interface and data contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is an imported helper module rather than a standalone command. Its public functions are exercised by the parent workflow scripts.
Inputs must exist before execution. Outputs are written to the CLI destinations or
to the path constants/defaults documented above and in the parser. Preserve CRS,
transform, resolution, nodata, band/feature order, and class IDs between dependent
stages; those properties are part of the analytical data contract.

Adapting to another area of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regenerate selected configurations and seed outputs from the new AOI training data; do not mix summaries from different spatial splits.
Record the replacement AOI, acquisition dates, CRS, resolution, class mapping, random
seed, and software environment. Validate intermediate dimensions/statistics and inspect
final maps or tables before using them in analysis or publication.

Reproducible invocation
~~~~~~~~~~~~~~~~~~~~~~~
Import this helper from its parent workflow or an interactive check::

    import scripts.final_10seed_experiment.common.seed_utils
"""

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

