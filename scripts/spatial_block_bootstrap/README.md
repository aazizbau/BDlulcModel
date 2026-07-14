# Spatial Block Bootstrap Accuracy Assessment

This package creates spatial-bootstrap confidence intervals for three accuracy
figures without modifying the existing analysis or training scripts:

1. selected model-family comparison;
2. AE64 versus AE64 plus ten indices;
3. class-wise Producer's Accuracy, User's Accuracy, and F1-score for the
   validation-selected best model.

The zone-wise LULC-area plot is intentionally excluded.

## Method

The original extraction files do not store coordinates or block IDs in the
NPZ. Stage 02 therefore replays the deterministic sample-selection procedure
against the unchanged source rasters, retaining the local raster block row and
column used by the original 1 km split. The recovered label sequence must match
`y_test` exactly.

The selected training runs already contain test predictions from their saved
best checkpoints. These predictions are joined to the recovered block IDs in
their original row order. This is equivalent to re-running inference but avoids
an unnecessary second prediction pass. The validation stage requires the sum
of all block confusion matrices to equal each saved test confusion matrix in
all 100 cells before bootstrapping is allowed.

For each replicate, the original number of test blocks is sampled with
replacement. Their confusion matrices are summed and pixel-based metrics are
recalculated from the aggregate. The same saved bootstrap indices are used for
every model and feature set, providing paired spatial comparisons.

## Complete Run

Activate the project environment and run:

```bash
source .venv/bin/activate

python scripts/spatial_block_bootstrap/run_all_spatial_bootstrap.py \
    --output-root outputs/spatial_block_bootstrap \
    --bootstrap 5000 \
    --seed 42 \
    --add-title
```

Preview all commands without writing outputs:

```bash
python scripts/spatial_block_bootstrap/run_all_spatial_bootstrap.py \
    --output-root outputs/spatial_block_bootstrap \
    --bootstrap 5000 \
    --seed 42 \
    --add-title \
    --dry-run
```

## Individual Stages

```bash
python scripts/spatial_block_bootstrap/preparation/01_identify_selected_runs.py
python scripts/spatial_block_bootstrap/preparation/02_export_test_predictions_by_block.py
python scripts/spatial_block_bootstrap/preparation/03_create_block_confusion_matrices.py
python scripts/spatial_block_bootstrap/preparation/04_validate_block_confusion_matrices.py
python scripts/spatial_block_bootstrap/preparation/05_generate_shared_bootstrap_indices.py \
    --bootstrap 5000 \
    --seed 42

python scripts/spatial_block_bootstrap/bootstrap/10_run_model_comparison_bootstrap.py
python scripts/spatial_block_bootstrap/bootstrap/11_run_featureset_comparison_bootstrap.py
python scripts/spatial_block_bootstrap/bootstrap/12_run_bestmodel_classwise_bootstrap.py

python scripts/spatial_block_bootstrap/visualization/20_plot_model_comparison_spatial_ci.py \
    --add-title
python scripts/spatial_block_bootstrap/visualization/21_plot_featureset_comparison_spatial_ci.py \
    --add-title
python scripts/spatial_block_bootstrap/visualization/22_plot_bestmodel_classwise_spatial_ci.py \
    --add-title
python scripts/spatial_block_bootstrap/visualization/23_create_spatial_bootstrap_tables.py \
    --add-title
```

## Outputs

All products are written below `outputs/spatial_block_bootstrap/`:

- `metadata/`: frozen run selection, test-block inventory, settings, and the
  mandatory validation report;
- `test_predictions_by_block/`: selected test predictions with original block
  identity;
- `block_confusion_matrices/`: complete block-level 10 x 10 matrices;
- `bootstrap_indices/`: shared paired resampling indices;
- `bootstrap_distributions/`: replicate-level metrics;
- `summaries/`: CSV confidence summaries and an Excel workbook;
- `figures/`: three confidence-interval plots and three PNG tables.

Parquet is used when a Parquet engine is installed. Otherwise, large tables
are written as compressed `.csv.gz` files and read transparently by later
stages.

## Interpretation

These intervals quantify uncertainty associated with the spatial composition
of the independent test sample. They do not measure variation from random
model initialization, stochastic optimization, or repeated training.
