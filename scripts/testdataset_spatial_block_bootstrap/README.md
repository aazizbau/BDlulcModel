# Test-Dataset Spatial Block Bootstrap

This workflow is a separate test-performance-selected version of
`scripts/spatial_block_bootstrap`. It uses the same original 1 km test blocks,
paired block-resampling indices, metric formulas, validation checks, plots, and
table exports. It does not alter the validation-selected workflow or its
outputs.

## Selection Rules

- Read all 120 completed runs from
  `outputs/master_training_with_outputs/all_master_runs_long.csv`.
- For each model-family/feature-set combination, select the run with the
  highest test Overall Accuracy.
- For each model-family comparison, retain the better of its selected AE64 and
  AE64-plus-ten-index runs according to test Overall Accuracy.
- Use the following fixed MLP run for the final class-wise analysis and best
  overall model:

```text
mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3
```

This is the same checkpoint used to produce the final 2017, 2023, and 2024
LULC inference rasters.

## Complete Run

```bash
source .venv/bin/activate

python scripts/testdataset_spatial_block_bootstrap/run_all_testdataset_spatial_bootstrap.py \
    --output-root outputs/testdataset_spatial_block_bootstrap \
    --bootstrap 5000 \
    --seed 42 \
    --add-title
```

Dry run:

```bash
python scripts/testdataset_spatial_block_bootstrap/run_all_testdataset_spatial_bootstrap.py \
    --output-root outputs/testdataset_spatial_block_bootstrap \
    --bootstrap 5000 \
    --seed 42 \
    --add-title \
    --dry-run
```

The workflow first writes its own frozen selection to:

```text
scripts/testdataset_spatial_block_bootstrap/config/selected_runs.yaml
outputs/testdataset_spatial_block_bootstrap/metadata/selected_runs.csv
```

All test-selected figures, summaries, block predictions, bootstrap
distributions, and validation records are kept under:

```text
outputs/testdataset_spatial_block_bootstrap/
```

## Methodological Note

Because the test set is used to select the runs in this workflow, its reported
test performance is descriptive and may be optimistically biased. The original
validation-selected workflow remains the appropriate source for an unbiased
model-selection procedure. This test-selected version is useful for reproducing
the model choice used by the existing LULC inference products.
