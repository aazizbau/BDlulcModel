# Final 10-Seed Best-Configuration Experiment

This folder runs a repeated-seed experiment for the single selected best
configuration from each model family. It does not perform another
hyperparameter search.

The fixed seed list is:

```text
11, 22, 33, 44, 55, 66, 77, 88, 99, 110
```

## Workflow

1. Freeze selected best configs from the existing master training outputs.

```bash
python scripts/final_10seed_experiment/prepare_selected_configs.py \
    --best-runs-csv outputs/master_training_with_outputs/best_runs_by_group.csv \
    --output-dir scripts/final_10seed_experiment/configs
```

2. Run one model family.

```bash
python scripts/final_10seed_experiment/train_mlp_10seeds.py \
    --config scripts/final_10seed_experiment/configs/mlp_best.yaml \
    --output-root outputs/final_10seed_experiment
```

3. Run all model families.

```bash
python scripts/final_10seed_experiment/run_all_10seed_experiments.py \
    --output-root outputs/final_10seed_experiment
```

4. Merge completed seed runs.

```bash
python scripts/final_10seed_experiment/merge_10seed_results.py \
    --output-root outputs/final_10seed_experiment
```

5. Plot empirical error bars from the actual repeated-seed metrics.

```bash
python scripts/final_10seed_experiment/plot_10seed_errorbars.py \
    --metrics-csv outputs/final_10seed_experiment/combined/all_run_metrics.csv \
    --output-dir outputs/final_10seed_experiment/figures \
    --add-title
```

## Output Structure

```text
outputs/final_10seed_experiment/
├── experiment_manifest.json
├── CNN1D/
│   ├── config/selected_config.yaml
│   ├── seed_011/
│   └── ...
├── FTTransformer/
├── LightGBM/
├── MLP/
├── ResMLP/
├── XGBoost/
├── combined/
│   ├── all_run_metrics.csv
│   ├── all_confusion_matrices_long.csv
│   └── model_family_summary.csv
└── figures/
    ├── overall_accuracy_mean_sd.png
    ├── macro_f1_mean_sd.png
    ├── weighted_f1_mean_sd.png
    └── test_metrics_grouped_errorbars.png
```

