import os
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Global Style Settings ---
matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.linewidth"] = 1.5      
plt.rcParams["xtick.major.width"] = 1.5   
plt.rcParams["ytick.major.width"] = 1.5

# === Data Loading and Preprocessing ===
rawdata_path = DATA_DIR / "sites_24_metadata.csv"
rawdata2_path = DATA_DIR / "profiles_24_soilwater_root.csv"
rawdata = pd.read_csv(rawdata_path, encoding='gbk')
rawdata2 = pd.read_csv(rawdata2_path, encoding='gbk')

err = np.empty((0, 20)) 

for k in range(1, 45):
    site_sum = np.empty((0, 20))
    site = np.empty((0, 6))
    for i in range(rawdata2.shape[0]):
        if rawdata2.iloc[i, 0] == k:
            site = np.vstack((site, rawdata2.iloc[i, :].values))
            
    df = pd.DataFrame(site)
    for j in range(df.shape[0]):
        if df.iloc[j, 1] > 3.0:
            df['sum1'] = df.iloc[15:j + 1, :].sum(axis=0)
            site_sum_hang = pd.DataFrame(np.empty((1, 20)))
            
            # Target variable: DSWD_mm (y)
            site_sum_hang.iloc[0, 0] = (df.iloc[2, 6] - df.iloc[3, 6]) * 200
            site_sum_hang.iloc[0, 1] = df.iloc[j, 1] + 0.1 # depth_m
            site_sum_hang.iloc[0, 2] = df.iloc[4, 6] # RDWD_kg_m3
            site_sum_hang.iloc[0, 3] = df.iloc[5, 6] # FRLD_m_m3
            
            row_indices = np.where(rawdata.iloc[:, 0] == k)[0]
            site_sum_hang.iloc[0, 4:20] = rawdata.iloc[row_indices, 1:17].values
            
            site_sum = np.vstack((site_sum, site_sum_hang))
    err = np.vstack((err, site_sum))

err_df = pd.DataFrame(err).drop_duplicates()
x = err_df.iloc[:, 1:20]
y = err_df.iloc[:, 0]

# === Variable Renaming (per Subset mapping table) ===
new_feature_names = {
    1: 'depth_m',
    2: 'RDWD_kg_m3',
    3: 'FRLD_m_m3',
    4: 'species_code',
    5: 'lat',
    6: 'lon',
    7: 'stand_age_yr',
    8: 'MAP_original_mm',
    9: 'MAP_gridded_mm',
    10:'planting_density_ha',
    11:'root_deepening_rate_m_yr',
    12:'max_rooting_depth_m',
    13:'PET_mm',
    14:'clay_pct',
    15:'sand_pct',
    16:'silt_pct',
    17:'root_C_shallow_Mg_ha',
    18:'root_C_deep_Mg_ha',
    19:'SOC_deep_pct'
}

x.rename(columns=new_feature_names, inplace=True)

# Select subset environmental features (Subset Background Model)
selected_features = ['depth_m', 'species_code', 'stand_age_yr', 'MAP_gridded_mm', 'PET_mm', 'clay_pct', 'sand_pct', 'silt_pct']
x = x[selected_features]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Model training
rfr = RandomForestRegressor(random_state=42, n_estimators=60, max_features=4, 
                                min_samples_leaf=1, min_samples_split=2, max_depth=5)
rfr.fit(X_train, Y_train)

# SHAP computation
explainer = shap.TreeExplainer(rfr)
shap_values = explainer.shap_values(X_test)

# === Plotting logic (labels and styles preserved) ===
plt.figure(figsize=(9, 6), dpi=600)  

# 1. Beeswarm Plot (bottom layer)
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False, color_bar=True)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
ax1 = plt.gca() 

# 2. Bar Plot (top layer, dual axes)
ax2 = ax1.twiny()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
ax2.axhline(y=len(selected_features)+1, color='gray', linestyle='-', linewidth=1) 

for bar in ax2.patches:
    bar.set_alpha(0.2)

# Axis labels (unchanged)
ax1.set_xlabel('Shapley Value Contribution')
ax1.spines['left'].set_visible(True)
ax1.spines['left'].set_linewidth(1)
ax1.spines['left'].set_color('black')

ax2.set_xlabel('Mean Shapley Value (Feature Importance)')
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()
ax2.spines['top'].set_visible(True)
ax2.spines['top'].set_linewidth(1)
ax2.spines['top'].set_color('black')
ax2.set_xlim(0, 300) # Adjust range based on data magnitude

ax1.set_ylabel('Features', fontsize=14)
plt.tight_layout()

plt.show()