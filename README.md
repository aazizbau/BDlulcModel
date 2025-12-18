# BDlulcModel

Pixel-wise land use/land cover (LULC) modeling toolkit for the coastal districts of Bangladesh. The project streamlines the workflow for downloading Google AlphaEarth embeddings, aligning them with local training labels, and building classification models that highlight the dynamics of the coastal zone.

## Key capabilities
- scripted downloader for AlphaEarth embeddings over the Bangladesh coastal strip (`scripts/gee/download_alphaearth_embeddings.py`).
- opinionated project layout for raw data, processed tiles, labels, configs, experiments, and notebooks.
- ready-to-extend Python package skeleton (`src/`) that will host GEE utilities, data pipelines, feature engineering, model training, inference, and visualization code.

## Repository layout
```
BDlulcModel/
├── assets/                 # project-ready figures, maps, and media
├── configs/                # YAML/JSON configs for GEE, training, inference
├── data/                   # raw/processed rasters, labels, and ancillary layers
├── docs/                   # requirements, references, design notes
├── notebooks/              # exploratory & prototyping notebooks
├── scripts/                # CLI utilities (GEE downloaders, data prep, modeling)
├── src/                    # importable Python modules
├── tests/                  # regression/unit tests
└── README.md               # you are here
```
Each data subfolder contains `.gitkeep` placeholders so that the directory structure remains under version control even though the heavy artifacts themselves are ignored.

## Getting started
1. **Create/activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Authenticate with Google Earth Engine** (one-time on each machine):
   ```bash
   earthengine authenticate
   ```
4. **Download AlphaEarth embeddings** for the Bangladesh coastal AOI:
   ```bash
   python scripts/gee/download_alphaearth_embeddings.py \
       --project ee-your-project-id \
       --year 2024 \
       --output data/raw/embeddings/bd_coastal_alphaearth_2024.tif \
       --tile-width-km 30 --tile-height-km 30 --tile-overlap-km 2
   ```
   Adjust the year, tile size, overlap, CRS, or pixel scale as needed for downstream experimentation.

## Data organization tips
- Drop raw AlphaEarth rasters (GeoTIFF) under `data/raw/embeddings/` once downloaded.
- Store exported GEE tables or auxiliary rasters in `data/raw/gee_exports/` and `data/external/`.
- Keep training/validation labels in `data/labels/points/` (vector) and `data/labels/masks/` (rasters).
- Use `data/interim/` for temporary merges/reprojections, and `data/processed/` for final tiles, mosaics, and ML-ready feature stacks.

## Next steps
- Integrate coastal district shapefiles and reference labels into the `data/` tree.
- Add preprocessing notebooks or scripts to harmonize training tiles with the embeddings.
- Implement feature extraction, modeling, and evaluation pipelines under `src/` and `scripts/`.
- Document modeling experiments in `experiments/` and log outputs in `logs/`.

Contributions and suggestions are welcome—open an issue or submit a PR once the repo is online.
