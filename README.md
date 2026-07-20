# BDlulcModel

Reproducible geospatial machine-learning workflow for ten-class land use and
land cover (LULC) mapping and change analysis in coastal Bangladesh. The project
combines 64-dimensional Google AlphaEarth embeddings with ten Sentinel-2
spectral indices, spatially separated training/validation/test samples, six
classification model families, blockwise inference, and spatial-bootstrap
uncertainty analysis.

The repository contains the complete processing logic used to produce coastal
LULC maps for 2017, 2023, and 2024. It can also be adapted to another area of
interest (AOI), provided that the replacement data are harmonized to one grid
and the model is retrained using labels from that AOI.

## Workflow Overview

```text
AOI and reference vectors
        |
        +-- AlphaEarth annual embeddings (64 bands)
        +-- Sentinel-2 Oct-Dec imagery (6 bands)
        +-- interpreted ten-class training labels
        |
        v
Mosaic, clip, reproject, resample, and align rasters
        |
        +-- AE64 feature set
        +-- AE64 + 10 Sentinel-2 indices feature set
        |
        v
1 km spatial-block train/validation/test sample extraction
        |
        v
Grid search: CNN1D, FTTransformer, LightGBM, MLP, ResMLP, XGBoost
        |
        v
Validation-based model selection and repeated/spatial uncertainty analysis
        |
        v
Blockwise annual inference, accuracy assessment, and 2017-2024 change analysis
        |
        v
Maps, tables, Sankey diagrams, parcel summaries, and thesis figures
```

## Scientific Data Contract

Downstream scripts assume the following project conventions. When transferring
the workflow to another AOI, update the values deliberately rather than merely
renaming files.

| Property | Current project convention |
|---|---|
| Geographic AOI definition | EPSG:4326 coordinates/vectors |
| Analysis CRS | EPSG:32646 (WGS 84 / UTM zone 46N) |
| Analysis resolution | 10 m |
| AlphaEarth predictors | 64 ordered embedding bands |
| Sentinel-2 indices | NDVI, EVI, MSAVI, NDMI, NDWI, NDPI, NDBI, BSI, NIRv, AWEI-SH |
| Combined feature order | `ae_01`...`ae_64`, followed by the ten indices above |
| LULC labels | Integer class IDs 1-10; label nodata is 0 |
| Spatial split | Deterministic 1 km blocks; default 70% train, 15% validation, 15% test |
| Extraction seed | 42 unless explicitly overridden |
| Training normalization | Mean and standard deviation calculated from training samples only |
| Continuous-feature nodata | AlphaEarth 0; aligned indices typically -9999 |
| Categorical resampling | Nearest neighbour |

The ten classes are:

1. Urban / Institutional Built-up
2. Rural Settlement (Homestead Vegetation)
3. Transport & Coastal Embankments
4. Cropland (All Crop Intensities)
5. Tree-based Agroforestry & Orchard
6. Aquaculture & Inland Ponds
7. Canals & Drainage Network
8. Rivers & Estuarine Channels
9. Mangrove Forest
10. Bare / Exposed Coastal Land

For another region, a different projected CRS may be more appropriate. Use a
local equal-distance or UTM CRS for 10 m alignment, block construction, distance,
and area calculations. All predictors, labels, and inference rasters must share
the same CRS, affine transform, width, height, resolution, and pixel origin.

## Repository Layout

```text
BDlulcModel/
|-- assets/                AOI vectors, training labels, palettes, map symbols
|-- configs/               AOI and workflow configuration files
|-- data/
|   |-- raw/               immutable downloads and source tiles
|   |-- interim/           mosaics, clipped bands, aligned indices
|   `-- processed/         ML-ready features, references, and NPZ samples
|-- logs/                  timestamped execution logs
|-- outputs/               inference, analysis tables, figures, bootstrap results
|-- runs/                  trained checkpoints and run-level metrics
|-- scripts/               processing, training, inference, analysis, visualization
|-- shell_scripts/         end-to-end and multi-stage orchestration
|-- src/                   reusable importable helpers
|-- requirements.txt       Python dependencies
`-- README.md
```

Large rasters, checkpoints, and generated outputs may be ignored by Git. A
reproduction therefore requires either regenerating them from the source data or
obtaining the archived data products separately.

## 1. Install the Environment

Run all commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

System GDAL libraries may be required before installing the Python `GDAL`
package. On Ubuntu/Debian, a typical setup is:

```bash
sudo apt update
sudo apt install gdal-bin libgdal-dev
```

Confirm the main tools:

```bash
python --version
gdalinfo --version
python -c "import rasterio, geopandas, torch, lightgbm, xgboost; print('OK')"
```

For GPU training, install a PyTorch build compatible with the local CUDA driver.
CPU execution is supported by most scripts but is substantially slower for model
training and full-coast inference.

## 2. Configure Earth Engine

Authenticate once and provide an Earth Engine project with export permission:

```bash
earthengine authenticate
export GEE_PROJECT_ID="your-earth-engine-project-id"
```

The current bounding-box AOI is stored in
[`configs/bd_coastal_aoi.yaml`](configs/bd_coastal_aoi.yaml). For a new AOI:

1. Copy the YAML and change its name and corner coordinates.
2. Replace the boundary vectors under `assets/maps/`.
3. Update any script with a hard-coded Bangladesh bounding box.
4. Use a new raw/interim/output directory so products from different AOIs are
   never mixed.

Inspect every script before execution:

```bash
python scripts/gee/download_alphaearth_embeddings.py --help
python scripts/s2_download/download_s2_octdec.py --help
```

## 3. Download AlphaEarth Embeddings

Download one year directly:

```bash
python scripts/gee/download_alphaearth_embeddings.py \
    --year 2023 \
    --project "$GEE_PROJECT_ID" \
    --output data/raw/embeddings/bd_coastal_alphaearth_2023.tif \
    --crs EPSG:4326 \
    --scale 10 \
    --tile-width-km 2.5 \
    --tile-height-km 2.5 \
    --tile-overlap-km 0.1
```

The repository also provides a logged 2023 acquisition/mosaic pipeline:

```bash
PROJECT="$GEE_PROJECT_ID" bash shell_scripts/run_alphaearth_2023_pipeline.sh
```

Before mosaicking, verify tile coverage with
`scripts/gee/check_alphaearth_tiles.py`. Missing-tile scripts are available for
the full coast and four training upazilas. Do not continue if the tile inventory
contains unexplained gaps.

## 4. Download Sentinel-2 Seasonal Imagery

The study uses October-December imagery for 2017, 2023, and 2024. The runner
downloads B02, B03, B04, B08, B11, and B12 and supports resume behavior:

```bash
PROJECT="$GEE_PROJECT_ID" \
AOI="configs/bd_coastal_aoi.yaml" \
bash shell_scripts/run_s2_octdec_download_multiyear_multiband.sh
```

For one custom request, use:

```bash
python scripts/s2_download/download_s2_octdec.py --help
```

Review the script options for year, band, AOI, cloud score, scale, CRS, output
location, and resume behavior. Keep original downloads under `data/raw/`.

## 5. Mosaic, Clip, and Harmonize Sentinel-2

Mosaic tiles with Rasterio or GDAL:

```bash
python scripts/sentinel/mosaic_sentinel_tiles.py --help
python scripts/sentinel/mosaic_sentinel_tiles_gdal.py --help
```

Clip all required years/bands to the solid coastal boundary and EPSG:32646:

```bash
bash scripts/clipping/run_clip_bdcoastalsolidUTM46_s2_octdec_multiyear_multiband.sh
```

For a new AOI, replace `assets/maps/bd_coastal_map_solid_gp.gpkg`, select an
appropriate projected CRS, and review continuous-band resampling in the clipping
script. Validate each output with `gdalinfo` before calculating indices.

## 6. Calculate the Ten Sentinel-2 Indices

Calculate all ten indices for all configured years:

```bash
bash scripts/s2_indices/run_calculate_s2_10indices_multiyear.sh
```

Each index also has an independent script under `scripts/s2_indices/`, for
example:

```bash
python scripts/s2_indices/make_ndvi_image.py --help
python scripts/s2_indices/make_awei_sh_image.py --help
```

The index rasters must inherit a common valid-data mask and be aligned before
feature stacking. Check scale factors and nodata handling if using Sentinel-2
data from a different provider.

## 7. Prepare Training Labels

The current training labels come from four upazilas: Manpura, Betagi, Amtali,
and Bamna. Each source vector requires valid `class10_id` and `class10_name`
fields. Rasterize the labels at 10 m:

```bash
for upazila in manpura betagi amtali bamna; do
    python scripts/labels/make_training_label_rastertif.py \
        --upazila "$upazila" \
        --out-dir assets/training_labels \
        --resolution 10 \
        --crs EPSG:32646
done
```

For another AOI, replace the label-vector mapping in the label script or expose
equivalent paths through its CLI. Confirm class frequencies before extraction:

```bash
python scripts/data_check/check_landclass_frequency.py --help
python scripts/data_check/inspect_gpkg_attributes.py --help
```

## 8. Align Features and Extract Spatial Samples

Prepare Float32 AlphaEarth data and align all indices to the authoritative
AlphaEarth grid. The training extraction scripts expect the four label rasters
and aligned predictors.

AE64-only dataset:

```bash
python scripts/training/extract_ae64_samples_trainvaltest.py \
    --ae data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif \
    --output data/processed/training/ae64_samples_4upazila_2023_trainvaltest.npz \
    --max-per-class 300000 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --block-size-m 1000 \
    --seed 42
```

AE64 plus ten indices dataset:

```bash
python scripts/training/extract_ae64_plus10indices_samples_trainvaltest.py \
    --ae data/interim/bd_coastal_fourupazila_alphaearth_2023_mosaic_f32.tif \
    --interim-dir data/interim/ae_aligned_indices_2023 \
    --output data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz \
    --max-per-class 300000 \
    --val-frac 0.15 \
    --test-frac 0.15 \
    --block-size-m 1000 \
    --seed 42
```

The split is performed by spatial block, not by randomly dividing individual
pixels. Normalization statistics are calculated from training samples only.
Validate both NPZ files before training:

```bash
python scripts/training/check_ae64_samples_trainvaltest_npz.py --help
python scripts/training/check_ae64_plus10indices_npz.py --help
```

For another AOI, reconsider block size, sample budgets, class balance, and split
fractions in relation to landscape scale and label coverage. Keep the same seed
when exact reproducibility is required.

## 9. Train and Tune the Models

Six model families are implemented for both feature sets:

- MLP
- ResMLP
- CNN1D
- FTTransformer
- LightGBM
- XGBoost

Run one model directly by following the complete example at the top of its
training script. For example:

```bash
python scripts/training/train_mlp_ae64_plus10indices_from_npz.py \
    --data data/processed/training/ae64_plus10indices_samples_4upazila_2023_trainvaltest.npz \
    --outdir runs/mlp_example \
    --hidden 512 256 \
    --dropout 0.3 \
    --batch-size 4096 \
    --epochs 100 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --patience 15 \
    --label-smoothing 0.05 \
    --scheduler \
    --device cuda \
    --seed 42
```

To run every configured grid and aggregate the results:

```bash
bash shell_scripts/run_master_training_with_outputs.sh
```

This command is computationally expensive. It launches 12 grid runners and can
train up to 120 configurations. Existing completed runs are resumed/skipped
according to the runner logic. Inspect
`outputs/master_training_with_outputs/logs/` and the run registry after
completion.

Model selection should use validation performance. The test set is reserved for
final reporting and must not be used repeatedly to tune hyperparameters.

## 10. Repeated-Seed and Spatial Uncertainty Analysis

Prepare and run the final ten-seed experiment:

```bash
python scripts/final_10seed_experiment/prepare_selected_configs.py --help
python scripts/final_10seed_experiment/run_all_10seed_experiments.py --help
python scripts/final_10seed_experiment/merge_10seed_results.py --help
python scripts/final_10seed_experiment/plot_10seed_errorbars.py --help
```

Run the validation-selected spatial block bootstrap for inferential model
comparison:

```bash
python scripts/spatial_block_bootstrap/run_all_spatial_bootstrap.py \
    --output-root outputs/spatial_block_bootstrap \
    --bootstrap 5000 \
    --seed 42 \
    --add-title
```

The separate test-selected workflow reproduces descriptive thesis plots and the
MLP checkpoint used for inference:

```bash
python scripts/testdataset_spatial_block_bootstrap/run_all_testdataset_spatial_bootstrap.py \
    --output-root outputs/testdataset_spatial_block_bootstrap \
    --bootstrap 5000 \
    --seed 42 \
    --add-title
```

Because the second workflow uses test performance during model selection, its
intervals are descriptive and may be optimistically biased. Use the
validation-selected workflow for methodologically unbiased model selection.

## 11. Prepare Annual Inference Inputs

For each target year, convert AlphaEarth to the inference grid and align the ten
indices:

```bash
for year in 2017 2023 2024; do
    python scripts/inference/make_ae64_ready_utm46_f32.py \
        --year "$year" \
        --overwrite

    python scripts/inference/make_align_10indices_with_AEGrid.py \
        --year "$year"
done
```

Validate the complete feature contract before classification:

```bash
python scripts/inference/validate_data_consistency_for_inference_ready.py \
    --years 2017 2023 2024 \
    --output-csv outputs/inference_ready_validation.csv
```

Do not run inference when any CRS, transform, dimension, nodata, band-count, or
feature-order check fails.

## 12. Run Annual LULC Inference

The project inference checkpoint is:

```text
runs/mlp_ae64plus10idx_h512-256_do03_lr1e3_bs4096_v3/best_model.pt
```

Run annual blockwise inference:

```bash
for year in 2017 2023 2024; do
    python scripts/inference/make_inference_with_bestmodel.py \
        --year "$year" \
        --device cuda \
        --output-root outputs/inference
done
```

The script writes class, confidence, and uncertainty rasters plus analysis-ready
CSV inventories. A new AOI requires a newly trained checkpoint with the exact
same feature order as its prepared inference stack.

## 13. Create Change and Parcel Products

Create pixel-level change products from the 2017 and 2024 classifications:

```bash
python scripts/inference/make_change_lulc_analysis_2017vs2024.py \
    --class-2017 outputs/inference/2017/lulc_class_2017.tif \
    --class-2024 outputs/inference/2024/lulc_class_2024.tif \
    --uncertainty-2017 outputs/inference/2017/uncertainty_2017.tif \
    --uncertainty-2024 outputs/inference/2024/uncertainty_2024.tif \
    --zone-map assets/maps/bd_coastal_zones.gpkg \
    --output-dir outputs/inference/change_analysis
```

Parcel workflows are under `scripts/parcels/` and parcel-change summaries under
`scripts/analysis/`. Replace parcel vectors, upazila identifiers, projected CRS,
and output paths for a new study region.

## 14. Generate Maps, Tables, and Thesis Figures

Every figure script under `scripts/visualization/` documents its required
rasters/vectors, output path, optional title behavior, and a complete command.
Typical products include:

```bash
python scripts/visualization/make_infer_lulc_map.py \
    --year 2017 \
    --add-title \
    --output-plot outputs/figures/bd_coastal_infer_lulc_2017.png \
    --output-csv outputs/figures/bd_coastal_infer_lulc_2017.csv

python scripts/visualization/visualize_lulc_change_sankey.py --help
python scripts/visualization/visualize_grouplulc_transition_2017vs2024.py --help
python scripts/visualization/visualize_sixlulc_transition_2017vs2024.py --help
```

Use `--add-title` only when a title should be embedded in the exported figure.
For thesis layouts where the caption provides the title, omit it.

## Script Documentation Standard

Every Python and shell script contains top-of-file documentation with:

- its role in the workflow;
- prerequisites and execution location;
- command-line options or import behavior;
- data-grid and output contracts;
- guidance for adapting paths, AOI, CRS, labels, and seeds;
- a worked command, usage example, or safe `--help`/import example.

Existing project-specific worked examples are retained. Before running any
script, read its header and inspect its parser:

```bash
sed -n '1,120p' scripts/path/to/script.py
python scripts/path/to/script.py --help
```

Some older scripts use path constants rather than CLI arguments. Adapt those
constants carefully and keep the change in version control for reproducibility.

## Reproduction Checklist

Before accepting a result, record and verify:

- Git commit and software environment;
- AOI geometry and administrative/reference vector versions;
- source product collection IDs, dates, cloud filters, and download logs;
- raster CRS, transform, resolution, dimensions, nodata, datatype, and band order;
- class-ID mapping and label provenance;
- spatial block size, split fractions, sampling budgets, and random seed;
- model family, hyperparameters, checkpoint, feature set, and normalization data;
- validation/test selection rule;
- bootstrap unit, number of blocks, replicates, confidence level, and seed;
- output inventories, validation reports, maps, and summary tables.

Never overwrite raw data or mix outputs from different AOIs, feature orders,
class mappings, or spatial splits.

## Troubleshooting

**Earth Engine authentication or quota failure**

Re-run `earthengine authenticate`, confirm `GEE_PROJECT_ID`, inspect export
permissions, reduce tile size, and use resume/missing-tile scripts.

**GDAL installation failure**

Install matching system GDAL development libraries before the Python package,
or use a geospatial Conda environment with compatible GDAL/Rasterio versions.

**Raster alignment failure**

Compare `gdalinfo` output for CRS, origin, pixel size, dimensions, and nodata.
Re-run alignment against one authoritative reference grid.

**CUDA out-of-memory error**

Reduce physical batch size, use gradient accumulation where supported, enable
AMP, or select CPU execution. Do not change feature order or labels.

**Interrupted training or download**

Use the documented resume behavior and timestamped logs. Confirm that a partial
output is complete and readable before allowing a downstream stage to use it.

## Project Scope

This repository was developed for an academic coastal-Bangladesh LULC study.
Its outputs should be interpreted with respect to the training-label coverage,
10 m mixed-pixel effects, class-specific confidence intervals, spatial sampling
design, and uncertainty documented by the repeated-seed and spatial-bootstrap
analyses.
