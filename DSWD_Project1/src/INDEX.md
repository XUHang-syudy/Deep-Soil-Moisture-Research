# Script Index

This directory contains all analysis scripts organized by phase.

## Directory Structure

| Directory | Phase | Description |
|-----------|-------|-------------|
| `01_optimization/` | 1 | Hyperparameter tuning |
| `02_sensitivity/` | 2 | Data proportion analysis |
| `03_evaluation/` | 3 | Model performance evaluation |
| `04_shap_global/` | 4 | SHAP beeswarm plots |
| `05_shap_dependence/` | 5 | SHAP dependence plots |
| `06_interaction/` | 6 | Variable interaction analysis |
| `07_importance/` | 7 | Depth-importance profiles |
| `08_mechanism/` | 8 | Root distribution analysis |
| `09_mapping/` | 9 | Study area map |

## Scripts by Directory

### 01_optimization/
- `param_tune_full_env.py` - Full-dataset Background Model (72 sites)
- `param_tune_subset_env.py` - Subset Background Model (24 sites)
- `param_tune_subset_root.py` - Subset Full-variable Model (24 sites + root traits)

### 02_sensitivity/
- `proportion_calc.py` - Training proportion calculations
- `proportion_viz.py` - Proportion sensitivity visualization

### 03_evaluation/
- `performance_calc.py` - Model performance metrics
- `performance_viz.py` - Prediction scatter/density plots

### 04_shap_global/
- `shap_summary_full_env.py` - SHAP beeswarm (Full-dataset)
- `shap_summary_subset_env.py` - SHAP beeswarm (Subset Environment)
- `shap_summary_subset_root.py` - SHAP beeswarm (Subset Full-variable)

### 05_shap_dependence/
- `dependence_full_env.py` - SHAP dependence (Full-dataset)
- `dependence_subset_env.py` - SHAP dependence (Subset Environment)
- `dependence_subset_root.py` - SHAP dependence (Subset Full-variable)

### 06_interaction/
- `interaction_matrix.py` - SHAP interaction matrix

### 07_importance/
- `importance_full_env.py` - Importance calculation (Full-dataset)
- `importance_subset_env.py` - Importance calculation (Subset Environment)
- `importance_subset_root.py` - Importance calculation (Subset Full-variable)
- `importance_viz.py` - Comparative visualization

### 08_mechanism/
- `root_distribution.py` - Root trait vertical profiles

### 09_mapping/
- `study_area_map.py` - Study area and site locations

## Paper Figure Mapping

See `README.md` in the root directory for the complete Figure-Script mapping table.
