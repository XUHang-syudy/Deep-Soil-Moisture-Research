# Reproducing the Results

This document provides step-by-step instructions to reproduce all figures and results in the paper.

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd dswd-rf

# 2. Create conda environment
conda env create -f environment.yml
conda activate dswd-rf

# 3. Run scripts by phase
```

## Environment Setup

### Option A: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate dswd-rf
```

### Option B: Using pip

```bash
pip install -r requirements.txt
```

## Data

All data files are located in the `data/` directory. See `data/README.md` for detailed column descriptions.

| File | Description |
|------|-------------|
| `sites_72_environmental.csv` | Site-level environmental variables (72 sites) |
| `profiles_72_soilwater.csv` | Vertical profile soil water content (72 sites) |
| `profiles_24_soilwater_root.csv` | Vertical profile with root traits (24 sites) |
| `sites_24_metadata.csv` | Metadata for 24-site subset |

## Reproducibility

All scripts use `random_state=42` for random number generation, ensuring identical results across different runs. This includes:
- Train/test data splitting
- Random Forest model initialization
- Bootstrap sampling in cross-validation

## Reproduce Figures

### Figure 3: Model Performance

```bash
python src/03_evaluation/performance_calc.py
python src/03_evaluation/performance_viz.py
```

Output: `results/figures/fig3_performance.png`

### Figure 4: SHAP Summary

```bash
python src/04_shap_global/shap_summary_full_env.py
python src/04_shap_global/shap_summary_subset_env.py
python src/04_shap_global/shap_summary_subset_root.py
```

Output: `results/figures/fig4a_shap_full_env.png`, `fig4b_shap_subset_env.png`, `fig4c_shap_subset_root.png`

### Figure 5-7: SHAP Dependence

```bash
python src/05_shap_dependence/dependence_full_env.py
python src/05_shap_dependence/dependence_subset_env.py
python src/05_shap_dependence/dependence_subset_root.py
```

### Figure 8: Interaction Matrix

```bash
python src/06_interaction/interaction_matrix.py
```

### Figure 9: Depth-Importance Profiles

```bash
python src/07_importance/importance_full_env.py
python src/07_importance/importance_subset_env.py
python src/07_importance/importance_subset_root.py
python src/07_importance/importance_viz.py
```

### Figure 1: Study Area Map (Optional)

> **Note**: This script requires PyGMT and GMT installation, which are platform-specific.
> The pre-generated map is already included in `results/figures/`.
> Only run this if you need to regenerate the map.

```bash
# Install PyGMT (requires GMT to be installed first)
# See: https://www.pygmt.org/latest/install.html
conda install -c conda-forge pygmt

python src/09_mapping/study_area_map.py
```

### Supplementary Figures

```bash
# Figure S1: Hyperparameter tuning
python src/01_optimization/param_tune_full_env.py
python src/01_optimization/param_tune_subset_env.py
python src/01_optimization/param_tune_subset_root.py

# Figure S2: Data proportion sensitivity
python src/02_sensitivity/proportion_calc.py
python src/02_sensitivity/proportion_viz.py
```

## Expected Runtime

| Phase | Scripts | Estimated Time |
|-------|---------|----------------|
| 01_optimization | `param_tune_*.py` | ~10 min |
| 02_sensitivity | `proportion_*.py` | ~5 min |
| 03_evaluation | `performance_*.py` | ~2 min |
| 04_shap_global | `shap_summary_*.py` | ~3 min |
| 05_shap_dependence | `dependence_*.py` | ~5 min |
| 06_interaction | `interaction_matrix.py` | ~3 min |
| 07_importance | `importance_*.py` | ~5 min |
| 08_mechanism | `root_distribution.py` | ~1 min |
| 09_mapping | `study_area_map.py` | ~1 min |

**Total**: ~35 minutes on a standard laptop.

## Troubleshooting

### SHAP Import Error

```bash
pip install shap --no-cache-dir
```

### Font Warnings

Matplotlib may show font warnings for Chinese characters. These can be safely ignored or resolved by installing the required fonts.

### Memory Issues

For large datasets, consider reducing the number of SHAP samples or using a machine with more RAM.

## Contact

For questions or issues, please contact the corresponding author or open an issue on GitHub.
