import os
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib import gridspec
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Global Plotting Configurations ---
red_transparent = to_rgba("red", alpha=0.15)
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

# === Step 1: Data Loading & Preprocessing ===
rawdata_path = DATA_DIR / "sites_72_environmental.csv"
rawdata2_path = DATA_DIR / "profiles_72_soilwater.csv"

rawdata = pd.read_csv(rawdata_path, encoding='gbk')
rawdata2 = pd.read_csv(rawdata2_path, encoding='gbk')
rawdata2_row = rawdata2.shape[0]

err = np.empty((0, 11))
for k in range(1, 73):
    site_sum = np.empty((0, 11))
    site = np.empty((0, 4))
    for i in range(rawdata2_row):
        if rawdata2.iloc[i, 0] == k:
            site = np.vstack((site, rawdata2.iloc[i, :]))
    df = pd.DataFrame(site)
    for j in range(df.shape[0]):
        if df.iloc[j, 1] > 3.0:
            df['sum1'] = df.iloc[15:j+1, :].sum(axis=0)
            site_sum_hang = pd.DataFrame(np.empty((1, 11)))
            site_sum_hang.iloc[0, 0] = (df.iloc[2, 4] - df.iloc[3, 4]) * 200
            site_sum_hang.iloc[0, 1] = df.iloc[j, 1] + 0.1 # depth_m
            site_sum_hang.iloc[0, 2:11] = rawdata.iloc[k - 1, 2:11]
            site_sum = np.vstack((site_sum, site_sum_hang))
    err = np.vstack((err, site_sum))

err_df = pd.DataFrame(err)
x = err_df.iloc[:, 1:9]
y = err_df.iloc[:, 0]

# --- Variable Renaming (per mapping table) ---
new_feature_names = {
    1: 'depth_m',
    2: 'MAP_mm',
    3: 'PET_mm',
    4: 'clay_pct',
    5: 'sand_pct',
    6: 'silt_pct',
    7: 'stand_age_yr',
    8: 'species_code',
}
x.rename(columns=new_feature_names, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# === Step 2: Model Training ===
rfr = RandomForestRegressor(random_state=42, n_estimators=60, max_features=4, 
                                min_samples_leaf=1, min_samples_split=2, max_depth=5)
rfr.fit(x_train, y_train)

# === Step 3: SHAP Value Computation ===
explainer = shap.TreeExplainer(rfr)
shap_test = explainer(x_test)
shap_df = pd.DataFrame(shap_test.values, columns=x_test.columns)
X_df = x_test.reset_index(drop=True)

# === Species Abbreviation Mapping ===
data = {
    'kinds': ['Malus pumila Mill.', 'Armeniaca vulgaris Lam.', 'Pinus tabulaeformis Carr.',
              'Populus tomentosa Carr.', 'Zanthoxylum bungeanum Maxim.', 'Robinia pseudoacacia Linn.',
              'Juglans regia Linn.', 'Sophora japonica L.', 'Ziziphus jujuba Mill.', 'Ulmus pumila L.',
              'Quercus wutaihanica Blume', 'Platycladus orientalis (Linn.) Franco' ],
    'code': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
}
df_species = pd.DataFrame(data)
df_species['abbreviation'] = df_species['kinds'].apply(lambda x: x.split()[0][:3])
abbreviations = df_species['abbreviation'].tolist()

# === Custom LOWESS Function ===
def plot_lowess_with_ci(x, y, ax, n_boot=1000, ci=95, color_line="red", frac=0.7):
    x, y = np.asarray(x), np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    x_grid = np.linspace(x.min(), x.max(), 200)
    boot_preds = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), len(x), replace=True)
        try:
            fit = lowess(y[idx], x[idx], frac=frac, return_sorted=True)
            order = np.argsort(fit[:, 0])
            xp_u, indices = np.unique(fit[:, 0][order], return_index=True)
            boot_preds.append(np.interp(x_grid, xp_u, fit[:, 1][order][indices]))
        except: continue
    boot_preds = np.array(boot_preds)
    if boot_preds.shape[0] < 5: return
    ax.fill_between(x_grid, np.percentile(boot_preds, (100-ci)/2, axis=0), 
                    np.percentile(boot_preds, 100-(100-ci)/2, axis=0), color=to_rgba(color_line, 0.2), zorder=1, lw=0)
    ax.plot(x_grid, np.median(boot_preds, axis=0), color=color_line, lw=1.8, zorder=2)

# === Colormap Configuration ===
cm_data = np.loadtxt(DATA_DIR / "bam.txt") 
bam_map = LinearSegmentedColormap.from_list('bam', cm_data)
vmin, vmax = 3, 25.5

# --- Combined Plotting (Tree Species, PET, MAP) ---
fig, axes = plt.subplots(3, 1, figsize=(9, 18), dpi=300)
fig.subplots_adjust(wspace=0.3)

# Update axis limit dictionary keys accordingly
yaxis_limits = {'species_code': (-300, 200), 'PET_mm': (-175, 150), 'MAP_mm': (-70, 100), 'stand_age_yr': (-600, 400)}
xaxis_limits = {'species_code': (100, 112), 'PET_mm': (875, 1225), 'MAP_mm': (350, 700), 'stand_age_yr': (0, 26)}
combined_features = ['species_code', 'PET_mm', 'MAP_mm']
labels = ['(a)', '(d)', '(g)']

for idx, feature in enumerate(combined_features):
    ax_main = axes[idx]
    x_val, shap_val = X_df[feature], shap_df[feature]
    color_feature = X_df['depth_m']

    scatter = ax_main.scatter(x_val, shap_val, s=10, c=color_feature, cmap=bam_map, alpha=0.8, vmin=vmin, vmax=vmax)
    ax_main.axhline(y=0, color='black', linestyle='-.', linewidth=1)
    ax_main.set_xlabel(feature)
    ax_main.set_ylabel("SHAP value")
    ax_main.tick_params(which='both', direction='in')

    if feature in yaxis_limits: ax_main.set_ylim(yaxis_limits[feature])
    if feature in xaxis_limits: ax_main.set_xlim(xaxis_limits[feature])
    
    if feature != 'species_code':
        plot_lowess_with_ci(x_val, shap_val, ax_main, color_line="red")
    else:
        ax_main.set_xlim(99.5, 111.5)
        ax_main.set_xticks(np.arange(100, 112, 1))
        ax_main.set_xticklabels(abbreviations, rotation=0)

    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
    sns.kdeplot(x=x_val, ax=ax_top, fill=True, color="#6A9ACE")
    ax_top.axis('off')
    ax_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
    sns.kdeplot(y=shap_val, ax=ax_right, fill=True, color="orange")
    ax_right.axis('off')
    ax_main.text(0.02, 0.95, labels[idx], transform=ax_main.transAxes, fontsize=22, va='top', ha='left')

plt.show()

# --- Individual Plots for Remaining Features ---
other_features = [col for col in x_test.columns if col not in combined_features]
for feature in other_features:
    x_val, shap_val = X_df[feature], shap_df[feature]
    fig = plt.figure(figsize=(9, 6), dpi=300)
    grid = gridspec.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    main_ax = fig.add_subplot(grid[1:, :-1])

    if feature != 'depth_m':
        color_f = X_df['depth_m']
        scatter = main_ax.scatter(x_val, shap_val, s=10, c=color_f, cmap=bam_map, alpha=0.8, vmin=vmin, vmax=vmax)
    else:
        scatter = main_ax.scatter(x_val, shap_val, s=10, c=shap_val, cmap='inferno', alpha=0.8)

    main_ax.axhline(y=0, color='black', linestyle='-.', linewidth=1)
    main_ax.set_xlabel(feature)
    main_ax.set_ylabel("SHAP value")
    if feature in yaxis_limits: main_ax.set_ylim(yaxis_limits[feature])
    if feature in xaxis_limits: main_ax.set_xlim(xaxis_limits[feature])
    if feature != 'species_code': plot_lowess_with_ci(x_val, shap_val, main_ax, color_line="red")

    top_ax = fig.add_subplot(grid[0, :-1], sharex=main_ax)
    sns.kdeplot(x=x_val, ax=top_ax, fill=True, color="#6A9ACE"); top_ax.axis('off')
    right_ax = fig.add_subplot(grid[1:, -1], sharey=main_ax)
    sns.kdeplot(y=shap_val, ax=right_ax, fill=True, color="orange"); right_ax.axis('off')
    plt.show()

print(f"✅ All renamed features have been processed.")