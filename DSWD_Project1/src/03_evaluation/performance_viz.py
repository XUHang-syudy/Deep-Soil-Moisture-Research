import os
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from matplotlib.colors import LinearSegmentedColormap

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ====== 1. Global Style Settings ======
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 16
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["axes.linewidth"] = 1.5      
plt.rcParams["xtick.major.width"] = 1.5   
plt.rcParams["ytick.major.width"] = 1.5

# ====== 2. Load Prediction Results for Three Models ======
file_paths = {
    'Full-dataset Background Model': RESULTS_DIR / "model_predictions" / "predictions_full_background.xlsx",
    'Subset Background Model': RESULTS_DIR / "model_predictions" / "predictions_subset_background.xlsx",
    'Subset Full-variable Model': RESULTS_DIR / "model_predictions" / "predictions_subset_full.xlsx"
}

# Try to load custom colormap
try:
    cm_data = np.loadtxt(DATA_DIR / "vik.txt") 
    batlow_map = LinearSegmentedColormap.from_list('batlow', cm_data)
except:
    batlow_map = 'viridis' # Fallback colormap if file is missing

models_data = {}
for model_name, path in file_paths.items():
    try:
        df = pd.read_excel(path)
    except:
        df = pd.read_csv(path)
    
    # --- Core Modification: Adapt to new column names ---
    # Create mapping to ensure compatibility with legacy or updated datasets
    col_mapping = {
        'depth_m': 'Depth(m)',
        'DSWD_value_mm': 'SWSD Value',
        'data_type': 'SWSD Type'
    }
    df = df.rename(columns=col_mapping)
    
    # After renaming, df columns are standardized to ['Depth(m)', 'SWSD Value', 'SWSD Type']
    observed = df[df['SWSD Type'].str.contains('Observed')][['Depth(m)', 'SWSD Value']].sort_values('Depth(m)').reset_index(drop=True)
    predicted = df[df['SWSD Type'].str.contains('Predicted')][['Depth(m)', 'SWSD Value']].sort_values('Depth(m)').reset_index(drop=True)
    
    if len(observed) != len(predicted):
        min_len = min(len(observed), len(predicted))
        observed = observed.iloc[:min_len]
        predicted = predicted.iloc[:min_len]
    
    paired = pd.DataFrame({
        'Depth(m)': observed['Depth(m)'],
        'Observed': observed['SWSD Value'],
        'Predicted': predicted['SWSD Value']
    })
    paired.dropna(inplace=True)
    models_data[model_name] = paired

# ====== 3. Define Custom Color Palette ======
custom_palette = {
    "Observed": (174/255, 198/255, 207/255, 0.8), 
    "Predicted": (255/255, 218/255, 185/255, 0.8) 
}

# ====== 4. Create 2x3 Subplot Layout ======
fig, axes = plt.subplots(2, 3, figsize=(27, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

model_names = ['Full-dataset Background Model', 'Subset Background Model', 'Subset Full-variable Model']
y_violin = (-500, 4000)
y_scatter = (-500, 3000)
x_scatter = (-500, 3000)

# ---------- First Row: Violin Plots ----------
for idx, model_name in enumerate(model_names):
    ax_v = axes[0, idx]
    data = models_data[model_name]
    
    # Construct plotting DataFrame
    obs_df = pd.DataFrame({'Value': data['Observed'], 'Type': 'Observed'})
    pred_df = pd.DataFrame({'Value': data['Predicted'], 'Type': 'Predicted'})
    df_plot = pd.concat([obs_df, pred_df], ignore_index=True)

    if not df_plot.empty:
        # Calculate quartiles for legend information
        quartiles = df_plot.groupby('Type')['Value'].quantile([0.25, 0.5, 0.75]).unstack()
        legend_labels = [
            f'{t}, Q1: {quartiles.loc[t, 0.25]:.1f}, Med: {quartiles.loc[t, 0.5]:.1f}, Q3: {quartiles.loc[t, 0.75]:.1f}'
            for t in quartiles.index
        ]
        
        # Split violin plot to compare distributions side-by-side
        sns.violinplot(x=[0]*len(df_plot), y='Value', hue='Type', data=df_plot,
                        split=True, inner="quartile", palette=custom_palette, ax=ax_v)
        
        handles, _ = ax_v.get_legend_handles_labels()
        ax_v.legend(handles=handles, labels=legend_labels, loc='upper right')
    
    # Labels remain unchanged: DSWU (mm)
    ax_v.set_ylabel("DSWU (mm)")
    ax_v.set_xlabel(model_name)
    ax_v.set_xticks([])
    ax_v.set_ylim(y_violin)
    ax_v.text(0.02, 0.95, f"({chr(97 + idx)})", transform=ax_v.transAxes, ha='left', va='top', fontsize=24)

# ---------- Second Row: Scatter Plots ----------
for idx, model_name in enumerate(model_names):
    ax_s = axes[1, idx]
    data = models_data[model_name]
    y_test = data['Observed'].values
    y_pred = data['Predicted'].values

    # Scatter plot color-mapped by soil depth
    scatter = ax_s.scatter(y_test, y_pred, c=data['Depth(m)'].values, cmap=batlow_map, 
                            alpha=0.7, vmin=3, vmax=25.5, edgecolor='black', linewidth=0.5)
    
    # 1:1 Reference Line
    lims = [max(x_scatter[0], y_scatter[0]), min(x_scatter[1], y_scatter[1])]
    ax_s.plot(lims, lims, '--', color='black', linewidth=1.5)

    # Linear regression fit
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    ax_s.plot(y_test, p(y_test), 'r--', linewidth=1.5)

    # Performance metrics calculation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    ax_s.text(0.50, 0.15, f'MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.2f}\ny = {z[0]:.2f}x + {z[1]:.2f}',
              transform=ax_s.transAxes, family='Arial', va='bottom')

    ax_s.set_xlim(x_scatter)
    ax_s.set_ylim(y_scatter)
    # Axis labels remain unchanged
    ax_s.set_xlabel('Observed DSWU (mm)')
    ax_s.set_ylabel('Predicted DSWU (mm)')
    ax_s.text(0.02, 0.95, f"({chr(97 + 3 + idx)})", transform=ax_s.transAxes, ha='left', va='top', fontsize=24)

# Colorbar label remains unchanged: Depth (m)
cax = fig.add_axes([0.1, 0.01, 0.8, 0.03])  
cbar = fig.colorbar(scatter, cax=cax, orientation='horizontal')
cbar.set_label('Depth (m)')

# ====== 5. Render Plot ======
plt.show()