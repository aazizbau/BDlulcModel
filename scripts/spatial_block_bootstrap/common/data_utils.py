"""Recover original test sample block IDs from deterministic extraction inputs."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from .constants import resolve_path
from .naming_utils import block_id


def load_npz_metadata(npz_path: str | Path) -> tuple[dict, np.ndarray]:
    path = resolve_path(npz_path)
    with np.load(path, allow_pickle=True) as archive:
        metadata = json.loads(str(archive["meta"].item()))
        y_test = archive["y_test"].astype(np.uint8)
    return metadata, y_test


def world_to_pixel(transform: rasterio.Affine, x: float, y: float) -> tuple[int, int]:
    col, row = ~transform * (x, y)
    return int(np.floor(col)), int(np.floor(row))


def block_assignment(row: int, col: int, block_px: int, seed: int) -> float:
    block_row = row // block_px
    block_col = col // block_px
    value = (block_row * 73856093) ^ (block_col * 19349663) ^ (seed * 83492791)
    value &= 0xFFFFFFFFFFFFFFFF
    return (value % 10_000_000) / 10_000_000.0


def split_name(value: float, validation_fraction: float, test_fraction: float) -> str:
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "val"
    return "train"


def _overlap_windows(
    label: rasterio.DatasetReader,
    ae: rasterio.DatasetReader,
) -> tuple[Window, Window, int, int]:
    left = max(label.bounds.left, ae.bounds.left)
    right = min(label.bounds.right, ae.bounds.right)
    bottom = max(label.bounds.bottom, ae.bounds.bottom)
    top = min(label.bounds.top, ae.bounds.top)
    if left >= right or bottom >= top:
        raise ValueError("Label and AlphaEarth rasters do not overlap.")

    label_c0, label_r0 = world_to_pixel(label.transform, left, top)
    label_c1, label_r1 = world_to_pixel(label.transform, right, bottom)
    ae_c0, ae_r0 = world_to_pixel(ae.transform, left, top)
    ae_c1, ae_r1 = world_to_pixel(ae.transform, right, bottom)

    label_c0, label_c1 = sorted((max(0, label_c0), min(label.width, label_c1)))
    label_r0, label_r1 = sorted((max(0, label_r0), min(label.height, label_r1)))
    ae_c0, ae_c1 = sorted((max(0, ae_c0), min(ae.width, ae_c1)))
    ae_r0, ae_r1 = sorted((max(0, ae_r0), min(ae.height, ae_r1)))

    height = min(label_r1 - label_r0, ae_r1 - ae_r0)
    width = min(label_c1 - label_c0, ae_c1 - ae_c0)
    if height <= 0 or width <= 0:
        raise ValueError("Invalid raster overlap dimensions.")
    return (
        Window(label_c0, label_r0, width, height),
        Window(ae_c0, ae_r0, width, height),
        int(height),
        int(width),
    )


def reconstruct_test_sample_metadata(
    npz_path: str | Path,
    chunk_size: int = 1024,
) -> pd.DataFrame:
    """Re-run deterministic sample selection while retaining test block identity."""
    metadata, expected_y_test = load_npz_metadata(npz_path)
    ae_path = resolve_path(metadata["ae_path"])
    upazilas = [str(value) for value in metadata["upazilas"]]
    label_paths = {
        key: resolve_path(value) for key, value in metadata["label_paths"].items()
    }
    index_paths = {
        key: resolve_path(value) for key, value in metadata.get("index_paths", {}).items()
    }
    index_order = list(index_paths)

    block_px = int(metadata["block_px"])
    seed = int(metadata["seed"])
    validation_fraction = float(metadata["val_frac"])
    test_fraction = float(metadata["test_frac"])
    max_remaining = {
        class_id: int(metadata["max_per_class"]) for class_id in range(1, 11)
    }
    label_nodata = int(metadata["label_nodata"])
    ae_nodata = float(metadata["ae_nodata"])
    index_nodata = float(metadata.get("index_nodata", -9999.0))
    rng = np.random.default_rng(seed)

    records: list[dict[str, object]] = []
    reconstructed_labels: list[int] = []

    with rasterio.open(ae_path) as ae:
        if ae.count != 64:
            raise ValueError(f"Expected 64 AlphaEarth bands in {ae_path}; found {ae.count}.")
        with ExitStack() as stack:
            indices = {
                name: stack.enter_context(rasterio.open(path))
                for name, path in index_paths.items()
            }

            for upazila in upazilas:
                with rasterio.open(label_paths[upazila]) as label:
                    if label.crs != ae.crs:
                        raise ValueError(f"CRS mismatch for {upazila}: {label.crs} vs {ae.crs}.")
                    label_full, ae_full, height, width = _overlap_windows(label, ae)

                    for row_offset in range(0, height, chunk_size):
                        chunk_height = min(chunk_size, height - row_offset)
                        for col_offset in range(0, width, chunk_size):
                            chunk_width = min(chunk_size, width - col_offset)
                            label_window = Window(
                                label_full.col_off + col_offset,
                                label_full.row_off + row_offset,
                                chunk_width,
                                chunk_height,
                            )
                            ae_window = Window(
                                ae_full.col_off + col_offset,
                                ae_full.row_off + row_offset,
                                chunk_width,
                                chunk_height,
                            )

                            labels = label.read(1, window=label_window)
                            valid = (
                                (labels != label_nodata)
                                & (labels >= 1)
                                & (labels <= 10)
                            )
                            if not valid.any():
                                continue

                            ae_values = ae.read(
                                list(range(1, 65)), window=ae_window
                            ).astype(np.float32)
                            if ae_nodata == 0.0:
                                ae_valid = np.all(ae_values != 0.0, axis=0)
                            else:
                                ae_valid = np.all(ae_values != ae_nodata, axis=0)
                            valid &= ae_valid & np.all(np.isfinite(ae_values), axis=0)

                            for name in index_order:
                                values = indices[name].read(1, window=ae_window).astype(
                                    np.float32
                                )
                                valid &= np.isfinite(values) & (values != index_nodata)
                            if not valid.any():
                                continue

                            rows, cols = np.where(valid)
                            order = np.arange(rows.size)
                            rng.shuffle(order)
                            rows = rows[order]
                            cols = cols[order]

                            for local_row, local_col in zip(rows, cols):
                                class_id = int(labels[local_row, local_col])
                                if max_remaining[class_id] <= 0:
                                    continue
                                global_row = int(label_window.row_off + local_row)
                                global_col = int(label_window.col_off + local_col)
                                split = split_name(
                                    block_assignment(
                                        global_row, global_col, block_px, seed
                                    ),
                                    validation_fraction,
                                    test_fraction,
                                )

                                if split == "test":
                                    current_block_row = global_row // block_px
                                    current_block_col = global_col // block_px
                                    x, y = label.transform * (
                                        global_col + 0.5,
                                        global_row + 0.5,
                                    )
                                    records.append(
                                        {
                                            "sample_id": len(records),
                                            "block_id": block_id(
                                                upazila,
                                                current_block_row,
                                                current_block_col,
                                            ),
                                            "block_row": current_block_row,
                                            "block_col": current_block_col,
                                            "source_upazila": upazila,
                                            "x": float(x),
                                            "y": float(y),
                                            "split": "test",
                                            "true_class_id": class_id,
                                        }
                                    )
                                    reconstructed_labels.append(class_id)

                                max_remaining[class_id] = max(
                                    0, max_remaining[class_id] - 1
                                )

                            if all(value <= 0 for value in max_remaining.values()):
                                break
                        if all(value <= 0 for value in max_remaining.values()):
                            break
                if all(value <= 0 for value in max_remaining.values()):
                    break

    reconstructed = np.asarray(reconstructed_labels, dtype=np.uint8)
    if not np.array_equal(reconstructed, expected_y_test):
        first_mismatch = None
        common = min(reconstructed.size, expected_y_test.size)
        mismatch = np.flatnonzero(reconstructed[:common] != expected_y_test[:common])
        if mismatch.size:
            first_mismatch = int(mismatch[0])
        raise RuntimeError(
            "Reconstructed test sample order does not match NPZ y_test. "
            f"Expected {expected_y_test.size}, reconstructed {reconstructed.size}, "
            f"first mismatch={first_mismatch}."
        )
    return pd.DataFrame(records)
