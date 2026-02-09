
import os
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib import gridspec
import matplotlib
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
# --- Global Style and Font Configurations ---
matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.linewidth"] = 1.5      
plt.rcParams["xtick.major.width"] = 1.5   
plt.rcParams["ytick.major.width"] = 1.5
red_transparent = to_rgba("red", alpha=0.15)

def plot_lowess_with_ci(x, y, ax, n_boot=1000, ci=95, color_line="red", frac=0.7):
    """Computes and plots a LOWESS trend line with bootstrapped confidence intervals."""
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    x_grid = np.linspace(x.min(), x.max(), 200)
    boot_preds = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), len(x), replace=True)
        try:
            fit = lowess(y[idx], x[idx], frac=frac, return_sorted=True)
            xp, yp = fit[:, 0], fit[:, 1]
            order = np.argsort(xp)
            xp_unique, indices = np.unique(xp[order], return_index=True)
            boot_preds.append(np.interp(x_grid, xp_unique, yp[order][indices]))
        except: continue
    boot_preds = np.array(boot_preds)
    if boot_preds.shape[0] < 5: return
    ax.fill_between(x_grid, np.percentile(boot_preds, (100 - ci) / 2, axis=0), 
                    np.percentile(boot_preds, 100 - (100 - ci) / 2, axis=0), 
                    color=to_rgba(color_line, 0.2), zorder=1, lw=0)
    ax.plot(x_grid, np.median(boot_preds, axis=0), color=color_line, lw=1.8, zorder=2)

# === Step 1: Data Loading & Preprocessing ===
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

# --- Feature Renaming ---
new_feature_names = {
    1: 'depth_m', 2: 'RDWD_kg_m3', 3: 'FRLD_m_m3', 4: 'species_code', 7: 'stand_age_yr', 
    9: 'MAP_gridded_mm', 13: 'PET_mm', 14: 'clay_pct', 15: 'sand_pct', 16: 'silt_pct'
}
x.rename(columns=new_feature_names, inplace=True)
selected_cols = ['depth_m', 'RDWD_kg_m3', 'FRLD_m_m3', 'species_code', 'stand_age_yr', 'MAP_gridded_mm', 'PET_mm', 'clay_pct', 'sand_pct', 'silt_pct']
x = x[selected_cols]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
rfr = RandomForestRegressor(random_state=42, n_estimators=60, max_features=4, max_depth=5)
rfr.fit(x_train, y_train)

explainer = shap.TreeExplainer(rfr)
shap_test = explainer(x_test)
shap_df = pd.DataFrame(shap_test.values, columns=x_test.columns)
X_df = x_test.reset_index(drop=True)

# === Species Abbreviation Mapping (Restored) ===
data_species = {
    'kinds': ['Malus pumila Mill.', 'Armeniaca vulgaris Lam.', 'Pinus tabulaeformis Carr.',
              'Populus tomentosa Carr.', 'Zanthoxylum bungeanum Maxim.', 'Robinia pseudoacacia Linn.',
              'Juglans regia Linn.', 'Sophora japonica L.', 'Ziziphus jujuba Mill.', 'Ulmus pumila L.',
              'Quercus wutaihanica Blume', 'Platycladus orientalis (Linn.) Franco' ],
    '标注': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
}
df_species = pd.DataFrame(data_species)
df_species['abbreviation'] = df_species['kinds'].apply(lambda x: x.split()[0][:3])
abbreviations = df_species['abbreviation'].tolist()

# Colormap
cm_data = np.loadtxt(os.path.join(r"C:\Users\Administrator\Desktop\DSWD_Project1\data", "bam.txt"))
bam_map = LinearSegmentedColormap.from_list('batlow', cm_data)
vmin, vmax = 3, 25.5

# Limits
yaxis_limits = {'species_code': (-300, 200), 'PET_mm': (-175, 150), 'MAP_gridded_mm': (-70, 100), 'stand_age_yr': (-600, 400), 'RDWD_kg_m3': (-200, 200), 'FRLD_m_m3': (-400, 400)}
xaxis_limits = {'species_code': (100, 112), 'PET_mm': (975, 1225), 'MAP_gridded_mm': (400, 700), 'stand_age_yr': (4, 26), 'RDWD_kg_m3': (-1500, 12500), 'FRLD_m_m3': (-15000, 275000)}

# ===================================
# Group 1: Biophysical Traits -> 1x2 Subplots
# ===================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
pair_features = ['RDWD_kg_m3', 'FRLD_m_m3']
pair_labels = ['(a)', '(b)']

for idx, feature in enumerate(pair_features):
    ax_main = axes[idx]
    x_val, shap_val = X_df[feature], shap_df[feature]
    scatter = ax_main.scatter(x_val, shap_val, s=10, c=X_df['depth_m'], cmap=bam_map, alpha=0.8, vmin=vmin, vmax=vmax)
    ax_main.axhline(y=0, color='black', linestyle='-.', lw=1)
    ax_main.set_xlabel(feature); ax_main.set_ylabel("SHAP value")
    if feature in yaxis_limits: ax_main.set_ylim(yaxis_limits[feature])
    if feature in xaxis_limits: ax_main.set_xlim(xaxis_limits[feature])
    
    formatter = ScalarFormatter(useMathText=False); formatter.set_scientific(True); formatter.set_powerlimits((-2, 3))
    ax_main.xaxis.set_major_formatter(formatter)
    plot_lowess_with_ci(x_val, shap_val, ax_main)

    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
    sns.kdeplot(x=x_val, ax=ax_top, fill=True, color="#6A9ACE"); ax_top.axis('off')
    ax_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
    sns.kdeplot(y=shap_val, ax=ax_right, fill=True, color="orange"); ax_right.axis('off')

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(scatter, cax=cbar_ax)
cb.set_label('depth_m')
plt.show()

# ===================================
# Group 2: Environmental Drivers -> 3x1 Subplots
# ===================================
combined_f = ['species_code', 'PET_mm', 'MAP_gridded_mm']
labels = ['(c)', '(f)', '(i)']
fig, axes = plt.subplots(3, 1, figsize=(10, 18), dpi=300)
for idx, feature in enumerate(combined_f):
    ax_main = axes[idx]
    x_val, shap_val = X_df[feature], shap_df[feature]
    scatter = ax_main.scatter(x_val, shap_val, s=10, c=X_df['depth_m'], cmap=bam_map, alpha=0.8, vmin=vmin, vmax=vmax)
    ax_main.axhline(y=0, color='black', linestyle='-.', lw=1)
    ax_main.set_xlabel(feature); ax_main.set_ylabel("SHAP value")
    if feature in yaxis_limits: ax_main.set_ylim(yaxis_limits[feature])
    if feature in xaxis_limits: ax_main.set_xlim(xaxis_limits[feature])
    
    if feature != 'species_code': 
        plot_lowess_with_ci(x_val, shap_val, ax_main)
    else:
        # Applying abbreviation mapping to species_code plot
        ax_main.set_xlim(99.5, 111.5)
        ax_main.set_xticks(np.arange(100, 112, 1))
        ax_main.set_xticklabels(abbreviations, rotation=0)

    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
    sns.kdeplot(x=x_val, ax=ax_top, fill=True, color="#6A9ACE"); ax_top.axis('off')
    ax_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
    sns.kdeplot(y=shap_val, ax=ax_right, fill=True, color="orange"); ax_right.axis('off')

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(scatter, cax=cbar_ax)
cb.set_label('depth_m')
plt.show()

# ===================================
# Group 3: Remaining Features (With Marginal KDEs)
# ===================================
other_f = [col for col in x_test.columns if col not in pair_features + combined_f]
for feature in other_f:
    x_val, shap_val = X_df[feature], shap_df[feature]
    fig = plt.figure(figsize=(9, 6), dpi=300)
    grid = gridspec.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    
    main_ax = fig.add_subplot(grid[1:, :-1])
    color_c = X_df['depth_m'] if feature != 'depth_m' else shap_val
    scatter = main_ax.scatter(x_val, shap_val, s=10, c=color_c, 
                               cmap=bam_map if feature != 'depth_m' else 'inferno', alpha=0.8)
    
    main_ax.axhline(y=0, color='black', linestyle='-.', lw=1)
    main_ax.set_xlabel(feature); main_ax.set_ylabel("SHAP value")
    if feature in yaxis_limits: main_ax.set_ylim(yaxis_limits[feature])
    if feature in xaxis_limits: main_ax.set_xlim(xaxis_limits[feature])
    plot_lowess_with_ci(x_val, shap_val, main_ax)

    # Added Marginal KDEs for Group 3
    top_ax = fig.add_subplot(grid[0, :-1], sharex=main_ax)
    sns.kdeplot(x=x_val, ax=top_ax, fill=True, color="#6A9ACE")
    top_ax.axis('off')
    right_ax = fig.add_subplot(grid[1:, -1], sharey=main_ax)
    sns.kdeplot(y=shap_val, ax=right_ax, fill=True, color="orange")
    right_ax.axis('off')

    cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('depth_m' if feature != 'depth_m' else 'SHAP value')
    
    plt.show()