import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Global Style Configurations ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.linewidth"] = 1.5      
plt.rcParams["xtick.major.width"] = 1.5   
plt.rcParams["ytick.major.width"] = 1.5

# === Step 1: Define File Paths ===
file_noroot    = RESULTS_DIR / "feature_importance" / "RF_importance_full_background.xlsx"
file_root      = RESULTS_DIR / "feature_importance" / "RF_importance_subset_full.xlsx"
file_noroot_24 = RESULTS_DIR / "feature_importance" / "RF_importance_subset_background.xlsx"

# === Step 2: Load Datasets ===
df_noroot = pd.read_excel(file_noroot)
df_root   = pd.read_excel(file_root)
df_noroot24 = pd.read_excel(file_noroot_24)

# === Step 3: Define Variable Columns (Updated to Scientific Names) ===

# Background model variables (Model 1 & Model 2)
vars_noroot = [
    'MAP_mm_mean', 'PET_mm_mean',
    'clay_pct_mean', 'sand_pct_mean', 'silt_pct_mean',
    'stand_age_yr_mean', 'species_code_mean'
]
stds_noroot = [
    'MAP_mm_std', 'PET_mm_std',
    'clay_pct_std', 'sand_pct_std', 'silt_pct_std',
    'stand_age_yr_std', 'species_code_std'
]

# Model 3 variables (with root traits)
# Note: Model 3 uses gridded precipitation (MAP_gridded_mm)
vars_root = [
    'MAP_gridded_mm_mean', 'PET_mm_mean',
    'clay_pct_mean', 'sand_pct_mean', 'silt_pct_mean',
    'stand_age_yr_mean', 'species_code_mean',
    'RDWD_kg_m3_mean', 'FRLD_m_m3_mean'
]
stds_root = [
    'MAP_gridded_mm_std', 'PET_mm_std',
    'clay_pct_std', 'sand_pct_std', 'silt_pct_std',
    'stand_age_yr_std', 'species_code_std',
    'RDWD_kg_m3_std', 'FRLD_m_m3_std'
]

# Model 2 uses similar variable naming as Model 1
vars_noroot24 = [
    'MAP_gridded_mm_mean', 'PET_mm_mean',
    'clay_pct_mean', 'sand_pct_mean', 'silt_pct_mean',
    'stand_age_yr_mean', 'species_code_mean'
]
stds_noroot24 = [
    'MAP_gridded_mm_std', 'PET_mm_std',
    'clay_pct_std', 'sand_pct_std', 'silt_pct_std',
    'stand_age_yr_std', 'species_code_std'
]

# === Step 4: Initialize 1x3 Subplot Layout ===
fig, axes = plt.subplots(1, 3, figsize=(27, 8), sharey=True)
plt.subplots_adjust(wspace=0.3)

# Plotting configuration tuple
plot_items = [
    (df_noroot, vars_noroot, stds_noroot, axes[0], '(a)'),
    (df_noroot24, vars_noroot24, stds_noroot24, axes[1], '(b)'),
    (df_root, vars_root, stds_root, axes[2], '(c)')
]

colors = plt.cm.tab10(np.linspace(0, 1, len(vars_root)))

# === Step 5: Execute Plotting Logic ===
for df, var_list, std_list, ax, tag in plot_items:
    depth = df['depth_m']
    for j, (m, s) in enumerate(zip(var_list, std_list)):
        mean = df[m]
        std = df[s]
        
        # Plot mean trend
        label_name = m.replace('_mean', '')
        ax.plot(depth, mean, label=label_name, color=colors[j], linewidth=2)
        # Fill standard deviation region
        ax.fill_between(depth, mean - std, mean + std, color=colors[j], alpha=0.2)
    
    ax.set_xlabel('Depth (m)')
    ax.tick_params(which='both', direction='out')
    
    # Subplot identifiers (a, b, c)
    ax.text(0.02, 0.95, tag, transform=ax.transAxes, fontsize=24, va='top', ha='left')
    
    if ax == axes[0]:
        ax.set_ylabel('Importance Value')
    
    # Adjust legend position to avoid overlap
    ax.legend(loc='upper right', fontsize=12, frameon=False)

# === Step 6: Layout Optimization and Display ===
plt.tight_layout()
plt.show()