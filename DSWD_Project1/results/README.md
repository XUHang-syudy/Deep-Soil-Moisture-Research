# Results Directory

This directory stores all model outputs and analysis results.

## Directory Structure

```
results/
├── feature_importance/     # Feature importance analysis results
├── figures/                # Generated figures
├── model_predictions/      # Model prediction outputs
└── tables/                 # Summary tables
```

## File Descriptions

### feature_importance/
- `RF_importance_full_background.xlsx` - Feature importance from full-dataset model
- `RF_importance_subset_background.xlsx` - Feature importance from subset environment model
- `RF_importance_subset_full.xlsx` - Feature importance from subset full-variable model

### model_predictions/
- `predictions_full_background.xlsx` - Predictions from full-dataset model
- `predictions_subset_background.xlsx` - Predictions from subset environment model
- `predictions_subset_full.xlsx` - Predictions from subset full-variable model

### figures/
Generated figures from analysis scripts.

### tables/
- `cv_performance_100runs.xlsx` - Cross-validation performance statistics (100 runs)

## Citation

> Wu, Q., et al. (2025). Nested interpretable machine learning reveals drivers 
> of deep soil water depletion. *Water Resources Research* (under review).
