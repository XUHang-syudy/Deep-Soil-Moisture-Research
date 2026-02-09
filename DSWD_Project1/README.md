# DSWD-RF: Nested Interpretable ML for Deep Soil Water Depletion

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains code and data for the nested interpretable machine learning framework used to quantify mechanistic drivers of Deep Soil Water Depletion (DSWD) on the Loess Plateau, China.

## Key Features

- **Nested Model Comparison**: Partitions proxy-driven vs. mechanism-driven contributions
- **SHAP Interpretation**: Quantifies feature importance and non-linear thresholds
- **Clear Structure**: Separate modules for data processing, modeling, and visualization

## Exploratory Notebook

`RF_DSWD.ipynb` contains the original exploratory analysis with full outputs. For formal reproduction, use the modular scripts in `src/`.

## Installation

```bash
# Clone repository
git clone <repository-url>
cd dswd-rf

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run scripts from the root directory. Ensure data is in the `/data` folder.

### Phase 1: Model Optimization & Robustness

Before full-scale analysis, optimize the Random Forest (RF) hyperparameters and test the model's sensitivity to training data proportions.

```bash
# Hyperparameter tuning for different model nests
python src/01_optimization/param_tune_full_env.py
python src/01_optimization/param_tune_subset_env.py
python src/01_optimization/param_tune_subset_root.py

# Sensitivity analysis for data proportions
python src/02_sensitivity/proportion_calc.py
python src/02_sensitivity/proportion_viz.py
```

### Phase 2: Model Evaluation

Evaluate the predictive accuracy of the nested models (Observed vs. Predicted).

```bash
python src/03_evaluation/performance_calc.py
python src/03_evaluation/performance_viz.py
```

### Phase 3: Mechanistic Interpretation (SHAP Analysis)

This is the core of the framework, quantifying the contribution of each driver and identifying non-linear responses.

```bash
# Global feature importance (Beeswarm plots)
python src/04_shap_global/shap_summary_full_env.py
python src/04_shap_global/shap_summary_subset_env.py
python src/04_shap_global/shap_summary_subset_root.py

# Individual variable dependence & interaction effects
python src/05_shap_dependence/dependence_full_env.py
python src/05_shap_dependence/dependence_subset_env.py
python src/05_shap_dependence/dependence_subset_root.py

python src/06_interaction/interaction_matrix.py
```

### Phase 4: Feature Importance

Calculate and visualize comparative feature importance across different depths.

```bash
python src/07_importance/importance_full_env.py
python src/07_importance/importance_subset_env.py
python src/07_importance/importance_subset_root.py
python src/07_importance/importance_viz.py
```

### Phase 5: Mechanism Verification (Root Traits)

Analyze the vertical distribution of root characteristics (RDWD, FRLD).

```bash
python src/08_mechanism/root_distribution.py
```

### Study Area Map

Generate the study area map with site locations.

```bash
python src/09_mapping/study_area_map.py
```

## Project Structure

```
.
├── data/                           # Raw and processed datasets
│   ├── sites_72_environmental.csv  # Site-level variables (72 sites)
│   ├── profiles_72_soilwater.csv   # Vertical profile SWC (72 sites)
│   ├── profiles_24_soilwater_root.csv  # Profile with root traits (24 sites)
│   └── sites_24_metadata.csv       # Metadata for 24-site subset
├── results/                        # Output directory
│   ├── feature_importance/         # RF feature importance
│   ├── figures/                    # Generated figures
│   ├── model_predictions/          # Model predictions
│   └── tables/                     # Numerical results
├── src/                            # Core Python scripts
│   ├── 01_optimization/            # Hyperparameter tuning
│   ├── 02_sensitivity/             # Data proportion analysis
│   ├── 03_evaluation/              # Model performance
│   ├── 04_shap_global/             # SHAP summary plots
│   ├── 05_shap_dependence/         # SHAP dependence plots
│   ├── 06_interaction/             # Variable interaction
│   ├── 07_importance/              # Depth-importance profiles
│   ├── 08_mechanism/               # Root distribution
│   └── 09_mapping/                 # Study area map
├── .gitignore                      # Git ignore rules
├── CITATION.cff                    # Citation metadata
├── environment.yml                 # Conda environment
├── LICENSE                         # CC BY 4.0
├── README.md                       # This file
├── REPRODUCE.md                    # Reproduction guide
└── requirements.txt                # Python dependencies
```

## Figure-Script Mapping

| Paper Figure | Script Path | Description |
|--------------|-------------|-------------|
| Figure 1a | `09_mapping/study_area_map.py` | Study area and site locations |
| Figure 2d-e | `08_mechanism/root_distribution.py` | Root trait vertical profiles |
| Figure 3 | `03_evaluation/performance_*.py` | Model performance plots |
| Figure 4a | `04_shap_global/shap_summary_full_env.py` | SHAP beeswarm (Full-dataset) |
| Figure 4b | `04_shap_global/shap_summary_subset_env.py` | SHAP beeswarm (Subset Environment) |
| Figure 4c | `04_shap_global/shap_summary_subset_root.py` | SHAP beeswarm (Subset Full-variable) |
| Figure 5-7 | `05_shap_dependence/dependence_*.py` | SHAP dependence plots |
| Figure 8 | `06_interaction/interaction_matrix.py` | SHAP interaction matrix |
| Figure 9 | `07_importance/importance_viz.py` | Depth-importance profiles |
| Figure S1 | `01_optimization/param_tune_*.py` | Hyperparameter tuning curves |
| Figure S2 | `02_sensitivity/proportion_viz.py` | Data proportion sensitivity |

## Results

The `results/` directory contains processed results:
- `figures/` - Generated paper figures
- `tables/` - Numerical results and metrics
- `model_predictions/` - Model predictions and SHAP values

Raw soil water content and root trait data are available upon reasonable request to the corresponding author.

## Citation

```bibtex
@article{wu2025dswd,
  title={Quantifying Mechanistic Drivers of Deep Soil Water Depletion 
         Using a Nested Interpretable Machine Learning Framework},
  author={Wu, Qifan and Xu, Hang and Su, Fuyuan and others},
  journal={Water Resources Research (under review)},
  year={2025}
}
```

## License

CC BY 4.0
