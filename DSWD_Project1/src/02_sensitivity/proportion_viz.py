import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ===== Load Dataset =====
df = pd.read_excel(RESULTS_DIR / "tables" / "cv_performance_100runs.xlsx")

# Configure global figure parameters for academic publication
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.linewidth"] = 1.5      # Axis frame linewidth
plt.rcParams["xtick.major.width"] = 1.5   # Major tick width
plt.rcParams["ytick.major.width"] = 1.5

# Calculate training set size from test_size
x = (1 - df["test_size"].to_numpy())

# ---- Extract Metrics Data ----
test_r2_mean = df["r2_test_mean"].to_numpy()
test_r2_std = df["r2_test_std"].to_numpy()
test_rmse_mean = df["rmse_test_mean"].to_numpy()
test_rmse_std = df["rmse_test_std"].to_numpy()
run_time_mean = df["run_time_mean"].to_numpy()
run_time_std = df["run_time_std"].to_numpy()

# ===== Define Axis Limits =====
r2_min, r2_max = 0.65, 1.00
rmse_min, rmse_max = 80, 310
runtime_min, runtime_max = 0, 0.10
size_min, size_max = 0, 3500

# ===== Tick Formatter =====
# Use 2 significant figures for Y-axis labels
fmt_sig2 = FuncFormatter(lambda v, pos: f"{v:.2g}")

# Initialize canvas with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# =========================
# Subplot (a): Run time analysis
# Visualizes computational efficiency (green) relative to training set size
# =========================
ax0 = axes[0]

# Plot mean run time with standard deviation as shaded area
ax0.plot(x, run_time_mean, color='green', label="Run time (s)")
ax0.fill_between(x, run_time_mean - run_time_std, run_time_mean + run_time_std,
                 color='green', alpha=0.2)

# Set Y-axis properties (Green)
ax0.set_ylabel("Run time (s)", color='green')
ax0.tick_params(axis='y', colors='green', width=2)
ax0.set_ylim(runtime_min, runtime_max)
ax0.set_yticks(np.linspace(runtime_min, runtime_max, 6))
ax0.yaxis.set_major_formatter(fmt_sig2)

# X-axis properties
ax0.set_xlabel("Training set size")
ax0.set_xlim(0, 1)

# Configure spine styles
ax0.spines['left'].set_color('green')
ax0.spines['left'].set_linewidth(2)
ax0.spines['bottom'].set_linewidth(2)
ax0.spines['top'].set_linewidth(2)
ax0.spines['right'].set_visible(2)

# Subplot legend
ax0.legend(loc='upper left', bbox_to_anchor=(0.30, 0.99))


# =========================
# Subplot (b): Predictive Accuracy (R² and RMSE)
# Dual Y-axis plot showing R² (Left, Black) and RMSE (Right, Blue)
# =========================
ax1 = axes[1]

# ---- Left Y-axis: R² (Coefficient of Determination) ----
ax1.plot(x, test_r2_mean, color='black', label="R²")
ax1.fill_between(x, test_r2_mean - test_r2_std, test_r2_mean + test_r2_std,
                 color='black', alpha=0.2)
ax1.set_xlabel("Training set size")
ax1.set_ylabel("R²", color='black')
ax1.tick_params(axis='y', colors='black', width=2)
ax1.set_ylim(r2_min, r2_max)
ax1.set_yticks(np.linspace(r2_min, r2_max, 6))
ax1.yaxis.set_major_formatter(fmt_sig2)
ax1.set_xlim(0, 1)
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['top'].set_linewidth(2)
ax1.spines['right'].set_visible(2)

# ---- Right Y-axis: RMSE (Root Mean Square Error) ----
ax1_r = ax1.twinx()
ax1_r.plot(x, test_rmse_mean, color='blue', label="RMSE")
ax1_r.fill_between(x, test_rmse_mean - test_rmse_std, test_rmse_mean + test_rmse_std,
                   color='blue', alpha=0.2)
ax1_r.set_ylabel("RMSE", color='blue')
ax1_r.tick_params(axis='y', colors='blue', width=2)
ax1_r.set_ylim(rmse_min, rmse_max)
ax1_r.set_yticks(np.linspace(rmse_min, rmse_max, 6))
ax1_r.spines["right"].set_color('blue')
ax1_r.spines['right'].set_linewidth(2)

# ---- Consolidated Legend for Subplot (b) ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
# Place legend at the upper center with 2 columns to avoid overlapping data
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper center', 
           bbox_to_anchor=(0.6, 0.99), 
           ncol=2, 
           frameon=True)

# =========================
# Subplot Identifiers (a) and (b)
# =========================
fig.text(0.09, 0.88, "(a)", ha='left', va='top', fontname='Arial', fontsize=24)
fig.text(0.57, 0.88, "(b)", ha='left', va='top', fontname='Arial', fontsize=24)

# =========================
# Layout Optimization and Export
# =========================
plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=2.5)
# plt.savefig(os.path.join("data", "combined_metrics.svg"), format='svg', dpi=300)
plt.show()