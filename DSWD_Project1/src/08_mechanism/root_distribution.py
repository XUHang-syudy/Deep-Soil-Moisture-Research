import os
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Global Style Configurations ---
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.5      # Axis frame linewidth
plt.rcParams["xtick.major.width"] = 1.5   # Major tick width
plt.rcParams["ytick.major.width"] = 1.5

# File path configuration
file_path = DATA_DIR / "raw_soilwater_measurements.xlsx"

# Sheet names and corresponding subplot titles
sheet_names = ['forestwater(3)', 'soilwater(3)', 'SWSD(3)',
               'Root Dry Weight Density (3)', 'Fine Root Length Density (3)']
titles = ['Forest', 'Cropland', 'DSWU', 'Root Dry Weight', 'Fine Root Length']

# Function to create a custom monochromatic gradient colormap
def create_monochrome_colormap(color_rgb, n=256, gamma=1.0):
    colors = []
    for i in np.linspace(0, 1, n):
        factor = i ** gamma
        r = (1 - factor) * 1.0 + factor * color_rgb[0]
        g = (1 - factor) * 1.0 + factor * color_rgb[1]
        b = (1 - factor) * 1.0 + factor * color_rgb[2]
        colors.append((r, g, b))
    return LinearSegmentedColormap.from_list('mono_cmap', colors)

# Colormap generation logic based on valid measurement counts
main_rgb = mcolors.to_rgb("#1f77b4")
base_cmap = create_monochrome_colormap(main_rgb, gamma=1.1)
brightness_factors = [1.0] * 5
colormap_start = 0.2
colormap_end = 1.2
cmaps = []
for factor in brightness_factors:
    colors_scaled = []
    for c in base_cmap(np.linspace(colormap_start, colormap_end, 256)):
        r, g, b = c[:3]
        r_new = min(r * factor, 1)
        g_new = min(g * factor, 1)
        b_new = min(b * factor, 1)
        colors_scaled.append((r_new, g_new, b_new))
    cmap_scaled = LinearSegmentedColormap.from_list(f'cmap_factor_{factor}', colors_scaled)
    cmaps.append(cmap_scaled)

# Initialize figure: 2 rows x 3 columns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(32, 36), sharey=True)
plt.subplots_adjust(wspace=0.15, hspace=0.10)
axes = axes.flatten()

# --- Iterate through the first five subplots (Boxplots) ---
for idx, (sheet, title, cmap) in enumerate(zip(sheet_names, titles, cmaps)):
    ax = axes[idx]
    # Load depth-profile data (Depth in first row, data in subsequent rows)
    df = pd.read_excel(file_path, sheet_name=sheet, header=None)
    depths = df.iloc[0, :].astype(float).values
    data = df.iloc[1:, :].astype(float).values.T
    adjusted_depths = np.arange(len(depths))

    # Outlier filtering (Percentile-based trimming)
    all_data = data.flatten()
    all_data = all_data[~np.isnan(all_data)]
    p1 = np.percentile(all_data, 0.1)
    p99 = np.percentile(all_data, 99.9)
    if sheet in ['Root Dry Weight Density (3)', 'Fine Root Length Density (3)']:
        data_trimmed = np.where((data > 0) & (data <= p99), data, np.nan)
    else:
        data_trimmed = np.where((data >= p1) & (data <= p99), data, np.nan)

    # Statistical calculations
    means = np.nanmean(data_trimmed, axis=1)
    mins = np.nanmin(data_trimmed, axis=1)
    maxs = np.nanmax(data_trimmed, axis=1)
    valid_counts = np.sum(~np.isnan(data_trimmed), axis=1)
    norm = mcolors.Normalize(vmin=1, vmax=72)
    colors = [cmap(norm(c)) for c in valid_counts]

    # Generate horizontal boxplots for each depth layer
    for i, (d, color) in enumerate(zip(data_trimmed, colors)):
        d_clean = d[~np.isnan(d)]
        if len(d_clean) > 0:
            bp = ax.boxplot(
                d_clean,
                positions=[adjusted_depths[i]],
                vert=False,
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor=color, color=color),
                medianprops=dict(color='red'),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker=''),
                showfliers=False
            )
            if i == 0:
                box_legend = bp['boxes'][0]
                whisker_legend = bp['whiskers'][0]
                cap_legend = bp['caps'][0]

    # Plot trend lines for medians and means
    medians = np.nanmedian(data_trimmed, axis=1)
    median_line, = ax.plot(medians, adjusted_depths, 'ro-', label='Median', linewidth=2, markersize=3)
    mean_point, = ax.plot(means, adjusted_depths, 'ko', label='Mean', markersize=3)
    # Add error bars/range lines
    for i in range(len(adjusted_depths)):
        ax.hlines(y=adjusted_depths[i], xmin=mins[i], xmax=maxs[i], color=colors[i], alpha=0.6)

    # Subplot labeling and styling
    ax.set_title(title, fontsize=22)
    ax.text(0.02, 0.98, f"({chr(97+idx)})", transform=ax.transAxes,
            fontsize=26, va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0))
    ax.tick_params(labelsize=22)
    ax.tick_params(axis='y', which='both', direction='out', length=6, width=1.2, color='black')
    
    # Configure X-axis (Top-positioned)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)
    
    ax.tick_params(which='both', bottom=False, top=True, left=True, right=False,
                   labelbottom=False, labelleft=True, direction='out')

    # Axis limits and labels based on variable type
    if sheet in ['forestwater(3)', 'soilwater(3)']:
        ax.set_xlim(0, 0.6)
        ax.set_xlabel('Soil Water Content (cm³/cm³)', fontsize=22)
    elif sheet == 'SWSD(3)':
        ax.set_xlim(-0.3, 0.3)
        ax.set_xlabel('Soil Water Content (cm³/cm³)', fontsize=22)
    elif sheet == 'Fine Root Length Density (3)':
        ax.set_xscale('log')
        ax.set_xlim(0.01, 1e8)
        ax.set_xlabel('Density (m/m³, log scale)', fontsize=22)
    elif sheet == 'Root Dry Weight Density (3)':
        ax.set_xscale('log')
        ax.set_xlim(0.01, 1e5)
        ax.set_xlabel('Density (g/m³, log scale)', fontsize=22)

    # Map adjusted indices back to physical depth labels (meters)
    integer_indices = [i for i, depth in enumerate(depths) if depth.is_integer()]
    integer_depths = [depths[i] for i in integer_indices]
    adjusted_integer_depths = [adjusted_depths[i] for i in integer_indices]
    ax.set_yticks(adjusted_integer_depths)
    ax.set_yticklabels([f"{int(d)}" for d in integer_depths], fontsize=22)
    
    if idx % 3 == 0:
        ax.set_ylabel('Depth (m)', fontsize=22)
    ax.invert_yaxis()

# --- Legend and Regression Insets ---
median_legend = Line2D([0], [0], color='black', linestyle='none', marker='s', markersize=3, label='Median')
handles = [mean_point, median_line, whisker_legend, box_legend]
labels = ['Mean', 'Median', 'Whisker(±1.5×IQR)', 'Box (IQR: Q1–Q3)']
axes[4].legend(handles=handles, labels=labels, loc='lower right', fontsize=20)

# Replace the 6th subplot with two nested regression plots
bbox = axes[5].get_position()
axes[5].remove()
gap = 0.009
h1 = (bbox.height - gap) * 0.5
h2 = h1

# Create sub-axes for Root Traits vs. Tree Age
ax1 = fig.add_axes([bbox.x0, bbox.y0 + h2 + gap, bbox.width, h1])
ax2 = fig.add_axes([bbox.x0, bbox.y0, bbox.width, h2])

# Load regression data
df_reg = pd.read_excel(file_path, sheet_name='RDWD with FRLD (2)')
x_reg = df_reg['Tree ages']
y1_reg = df_reg['root dry weight density']
y2_reg = df_reg['Fine root length density']

# Plot regressions (RDWD and FRLD)
sns.regplot(x=x_reg, y=y1_reg, ax=ax1, scatter_kws={'s': 65,'color': '#104680'}, line_kws={'color': 'red'}, ci=95)
sns.regplot(x=x_reg, y=y2_reg, ax=ax2, scatter_kws={'s': 65,'color': '#104680'}, line_kws={'color': 'blue'}, ci=95)

# Style regression subplots
ax1.set_xlabel('Tree Ages (year)', fontsize=22)
ax1.set_ylabel('Root Dry Weight Density (g/m³)', fontsize=20)
ax1.text(0.02, 0.95, '(f)', transform=ax1.transAxes, fontname='Arial', va='top', fontsize=26)
ax1.xaxis.set_label_position('top')
ax1.xaxis.tick_top()
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

ax2.set_ylabel('Fine Root Length Density (m/m³)', fontsize=22)
ax2.text(0.02, 0.95, '(g)', transform=ax2.transAxes, fontname='Arial', va='top', fontsize=26)
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()

# Function to annotate Pearson correlation statistics
def add_stats(ax, x, y):
    r_value, p_value = stats.pearsonr(x, y)
    n = len(x)
    ax.annotate(f'Pearson r = {r_value:.2f}\np = {p_value:.3f}\nn = {n}', 
                xy=(0.60, 0.85), xycoords='axes fraction', fontsize=22, verticalalignment='top')

add_stats(ax1, x_reg, y1_reg)
add_stats(ax2, x_reg, y2_reg)

# Global Colorbar for Boxplots
sm = ScalarMappable(cmap=cmaps[-1], norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Number of Valid Measurements', fontsize=22)
cbar.ax.tick_params(labelsize=22)

plt.show()