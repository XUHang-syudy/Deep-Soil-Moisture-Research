
import os
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set font family to Arial
plt.rcParams["font.family"] = "Arial"
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.linewidth"] = 1.5      # Axis frame thickness
plt.rcParams["xtick.major.width"] = 1.5   # Major tick thickness (X-axis)
plt.rcParams["ytick.major.width"] = 1.5   # Major tick thickness (Y-axis)

# Define file paths
rawdata_path = DATA_DIR / "sites_24_metadata.csv"
rawdata2_path = DATA_DIR / "profiles_24_soilwater_root.csv"
rawdata = pd.read_csv(rawdata_path, encoding='gbk')  # Load characteristic data
rawdata2 = pd.read_csv(rawdata2_path, encoding='gbk')  # Load SWSD data
rawdata2_row = np.size(rawdata2, 0)  # Calculate number of rows for X
rawdata2_col = np.size(rawdata2, 1)  # Calculate number of columns for X

err = np.empty((0, 20)) # Initialize container for processed data (20 columns)

# Loop through sites (ID 1 to 44)
for k in range(1, 45):
    site_sum = np.empty((0, 20))
    site = np.empty((0, 6))
    for i in range(0, rawdata2_row):
        if rawdata2.iloc[i, 0] == k:
            site = np.vstack((site, rawdata2.iloc[i, :].values))  # Convert to NumPy array
    
    site_row = np.size(site, 0)
    site_col = np.size(site, 1)
    df = pd.DataFrame(site)  # Convert NumPy array to DataFrame for processing
    
    for j in range(0, site_row):
        if df.iloc[j, 1] > 3.0:
            # Calculate cumulative sum for the specific depth profile
            df['sum1'] = df.iloc[15:j + 1, :].sum(axis=0)
            site_sum_hang = np.empty((1, 20))  # Temporary row container
            depth = df.iloc[j, 1]  # Extract current depth
            site_sum_hang = pd.DataFrame(site_sum_hang)
            
            # Feature calculation and integration
            site_sum_hang.iloc[0, 0] = (df.iloc[2, 6] - df.iloc[3, 6]) * 200  # Cumulative calculation
            site_sum_hang.iloc[0, 1] = df.iloc[j, 1] + 0.1  # Depth feature
            site_sum_hang.iloc[0, 2] = df.iloc[4, 6]  # Root dry weight density
            site_sum_hang.iloc[0, 3] = df.iloc[5, 6]  # Fine root length density
            
            # Retrieve site-specific variables from auxiliary data
            row_indices = np.where(rawdata.iloc[:, 0] == k)[0]
            site_sum_hang.iloc[0, 4:20] = rawdata.iloc[row_indices, 1:17].values
            
            site_sum = np.vstack((site_sum, site_sum_hang))
    err = np.vstack((err, site_sum))

err = pd.DataFrame(err)
C = err.drop_duplicates()
x = err.iloc[:, 1:20]  # Input features
y = -err.iloc[:, 0]  # Target variable (Inverted for DSWD)

# Feature Renaming
# Mapping original column indices to descriptive feature names
new_feature_names = {
    1: 'Depth(m)',
    2: 'RDWD', # Root Dry Weight Density
    3: 'FRLD', # Fine Root Length Density
    4: 'Tree Species',
    5: 'Latitude',
    6: 'Longitude',
    7: 'Tree Age(year)',
    8: 'P(mm yr-1)',
    9: 'Precipitation(mm)',
    10:'Planting density(plants/ha)',
    11:'Root deepening rate(m yr-1)',
    12:'Maximum rooting depth(m)',
    13:'PET(mm)',
    14:'Clay(%)',
    15:'Sand(%)',
    16:'Silt(%)',
    17:'root biomass C input in shallow soil(Mg ha-1)',
    18:'root biomass C input in deep soil(Mg ha-1)',
    19:'C% in deep soil'
}

# Apply column renaming
x.rename(columns=new_feature_names, inplace=True)

# Select subset of features for the "Subset Background Model"
x = x[['Depth(m)','Tree Species','Tree Age(year)','Precipitation(mm)','PET(mm)', 'Clay(%)', 'Sand(%)', 'Silt(%)']]

# Dataset splitting: 70% Training, 30% Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize 1x3 subplot figure
fig, axes = plt.subplots(1, 3, figsize=(27, 6))

# ===== Subplot 1: Grid Search for max_depth =====
param_grid = {'max_depth': np.arange(1, 11)}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)
results = grid_search.cv_results_
best_scores = np.sqrt(-results['mean_test_score']) # Convert Neg-MSE to RMSE
max_depths = results['param_max_depth'].data

axes[0].plot(max_depths, best_scores, marker='o', linestyle='-', color='blue', label='Max Depth')
axes[0].set_xlabel('Max Depth')
axes[0].set_ylabel('Best Score (RMSE)')
#axes[0].set_title('(a)')
axes[0].legend()
axes[0].set_xticks(np.arange(0, 11, 1))
# Annotate subplot (d)
axes[0].text(0.02, 0.95, '(d)', transform=axes[0].transAxes, fontsize=24, va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0))
axes[0].set_yticks(np.linspace(0, 450, 6))
axes[0].tick_params(which='both', bottom=True, top=False, left=True, right=False,
                    labelbottom=True, labelleft=True, direction='out')

# ===== Subplot 2: Grid Search for n_estimators =====
param_grid = {'n_estimators': np.arange(5, 300, 5), 'max_depth': [5]}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)
results = grid_search.cv_results_
best_scores = np.sqrt(-results['mean_test_score'])
n_estimators = results['param_n_estimators'].data

axes[1].plot(n_estimators, best_scores, marker='o', linestyle='-', color='green', label='Num Estimators')
axes[1].set_xlabel('Number of Estimators')
#axes[1].set_ylabel('Best Score (RMSE)')
#axes[1].set_title('(b)')
axes[1].legend()
# Annotate subplot (e)
axes[1].text(0.02, 0.95, '(e)', transform=axes[1].transAxes, fontsize=24, va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0))
axes[1].set_xticks(np.arange(0, 350, 50))
axes[1].set_yticks(np.linspace(120, 140, 6))
axes[1].tick_params(which='both', bottom=True, top=False, left=True, right=False,
                    labelbottom=True, labelleft=True, direction='out')

# ===== Subplot 3: Grid Search for min_samples_leaf / split / max_features =====
# Hyperparameter optimization for min_samples_leaf
param_grid1 = {'min_samples_leaf': np.arange(1, 11), 'max_depth': [5]}
rf1 = RandomForestRegressor(random_state=42)
grid_search1 = GridSearchCV(estimator=rf1, param_grid=param_grid1, cv=10, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search1.fit(x_train, y_train)
results1 = grid_search1.cv_results_
best_scores1 = np.sqrt(-results1['mean_test_score'])
min_samples_leaf = results1['param_min_samples_leaf'].data

# Hyperparameter optimization for min_samples_split
param_grid2 = {'min_samples_split': np.arange(2, 11), 'max_depth': [5]}
rf2 = RandomForestRegressor(random_state=42)
grid_search2 = GridSearchCV(estimator=rf2, param_grid=param_grid2, cv=10, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search2.fit(x_train, y_train)
results2 = grid_search2.cv_results_
best_scores2 = np.sqrt(-results2['mean_test_score'])
min_samples_split = results2['param_min_samples_split'].data

# Hyperparameter optimization for max_features
param_grid3 = {'max_features': np.arange(1, 9), 'max_depth': [5]}
rf3 = RandomForestRegressor(random_state=42)
grid_search3 = GridSearchCV(estimator=rf3, param_grid=param_grid3, cv=10, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search3.fit(x_train, y_train)
results3 = grid_search3.cv_results_
best_scores3 = np.sqrt(-results3['mean_test_score'])
max_features = results3['param_max_features'].data

axes[2].plot(min_samples_leaf, best_scores1, marker='o', linestyle='-', color='blue', label='Min Samples Leaf')
axes[2].plot(min_samples_split, best_scores2, marker='^', linestyle='-', color='green', label='Min Samples Split')
axes[2].plot(max_features, best_scores3, marker='s', linestyle='-', color='red', label='Max Features')
axes[2].set_xlabel('Min Samples Leaf/Min Samples Split/Max Features')
#axes[2].set_ylabel('Best Score (RMSE)')
#axes[2].set_title('(c)')
axes[2].legend()
axes[2].set_xticks(np.arange(0, 11, 1))
# Annotate subplot (f)
axes[2].text(0.02, 0.95, '(f)', transform=axes[2].transAxes, fontsize=24, va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0))
axes[2].set_yticks(np.linspace(100, 250, 6))
axes[2].tick_params(which='both', bottom=True, top=False, left=True, right=False,
                    labelbottom=True, labelleft=True, direction='out')

# ===== Layout Adjustment and Figure Export =====
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figS1b_hyperparam_subset_env.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "figS1b_hyperparam_subset_env.svg", format='svg', bbox_inches='tight')
plt.show()