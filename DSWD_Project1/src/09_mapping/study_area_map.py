import pygmt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# === Set working directory ===
os.chdir(PROJECT_ROOT / "data" / "gmt")

# === Create figure ===
fig = pygmt.Figure()
region = [99, 117, 33, 42]
projection = "M15c"

# === Load topographic background ===
grid = pygmt.datasets.load_earth_relief(resolution="01m", region=region)
pygmt.makecpt(cmap="white,255/255/255,250/250/250,240/240/240", series="-500/8000/1000", reverse=True)

fig.grdimage(
    grid=grid, 
    projection=projection,
    shading="-I+d+a0.01",
    cmap=True
)

# === Load point data ===
df_pts   = pd.read_csv("unique_points_with_count72.gmt", delim_whitespace=True, header=None, names=["lon", "lat", "number"])
df_roots = pd.read_csv("unique_points_with_count24.gmt", delim_whitespace=True, header=None, names=["lon", "lat", "number"])

df_pts = df_pts.apply(pd.to_numeric, errors='coerce')
df_roots = df_roots.apply(pd.to_numeric, errors='coerce')

size_mapping = {
    1: 0.12, 2: 0.14, 3: 0.16, 4: 0.18,
    5: 0.20, 8: 0.22, 10: 0.24, 14: 0.26
}
df_pts["size"] = df_pts["number"].map(size_mapping).fillna(0.12)
df_roots["size"] = df_roots["number"].map(size_mapping).fillna(0.12)

# === Plot actual data points ===
fig.plot(
    x=df_pts["lon"], y=df_pts["lat"],
    style="c", size=df_pts["size"],
    fill="#0072B2", pen="0.3p,black",
    label="Forest Sites"
)
fig.plot(
    x=df_roots["lon"], y=df_roots["lat"],
    style="c", size=df_roots["size"],
    fill="#D55E00", pen="0.3p,black",
    label="Cropland Sites"
)

# === Add schematic points for legend (triggering legend) ===
legend_counts = [1, 2, 3, 4, 5, 8, 10, 14]  # Number of points to display in the legend
for i, count in enumerate(legend_counts):
    size = size_mapping[count]
    fig.plot(
        x=[98.5], y=[44.5 - i * 0.3],   # Offset each circle downwards to avoid overlap
        style=f"c{size}c",
        fill="black",
        pen="0.3p,black",
        label=f"n = {count}"
    )
fig.plot(x=[98.5], y=[44.5], style="c0.22c", fill="#0072B2", pen="0.3p,black", label="Forest Sites")
fig.plot(x=[98.5], y=[44.5], style="c0.22c", fill="#D55E00", pen="0.3p,black", label="Cropland Sites")

# === Boundary layers and color unification ===
fig.plot(data="Isohyet.gmt", pen="1p,#4682B4", label="Rainfall Isolines")
# fig.plot(data="Maowusu_Desert.gmt", pen="1p,#A0522D", label="Maowusu Desert")
fig.plot(data="Loess_Plateau.gmt", pen="1p,#8B4513", label="Loess Plateau")
fig.plot(data="Loess_region.gmt", pen="2p,#000000", label="Loess Region")

# === Set map style, scale bar, and North arrow ===
with pygmt.config(
    FONT="16p,Helvetica,#4A4A4A",
    FONT_TAG="16p,Helvetica,#4A4A4A",
    FONT_LABEL="16p,Helvetica,#4A4A4A",
    FONT_ANNOT_PRIMARY="16p,Helvetica,#4A4A4A",
    MAP_FRAME_PEN="1p,#4A4A4A",
    MAP_TICK_LENGTH="0.2c"                # ✅ Set tick mark length
  # MAP_TICK_PEN="0.5p,#4A4A4A",
  # MAP_GRID_PEN="0.25p,#AAAAAA",
  # MAP_FRAME_TYPE="plain" 
):
    fig.basemap(region=region, projection=projection, frame=["WSen", "xa3f2+lx°E", "ya3f2+ly°N"])
    fig.basemap(rose="jTR+w1.5c+f1+l+o0.2c+pen0.5p,#4A4A4A")
    fig.basemap(map_scale="jBL+w400k+f+lKm+ar+o0.2/0.5c")

# === Add legend box ===
# fig.legend(position="JBR+jBR+o0.4c", box="+gwhite+p0.5p,#4A4A4A")

# === Insets ===
# === Inset 1: Overview map of China ===
with fig.inset(position="jTL+w3.0c+o0.01c", box="+p1p,black"):
    fig.coast(
        region=[70, 140, -5, 55],
        projection="M4.0c",
        # land="lightgray",
        land="white",
        water="white"
    )
    fig.plot(data="china_border.gmt", pen="0.1p,#4A4A4A")
    fig.plot(data="Loess_Plateau.gmt", pen="1p,#8B4513")
    fig.plot(data="bianjing.gmt", pen="0.1p,#4A4A4A")

# ===================================================================================
# === INSET 2: Enlarged South China Sea (positioned at jBR of main map) ===
# Note: This is an independent second inset, not nested.
# ===================================================================================
with fig.inset(position="x1.75c/6.25c+w1.0c+o0.2c", box="+p0.5p,black"): #,gray+gwhite
    fig.coast(
        region=[105, 125, 3, 25],  # South China Sea region
        projection="M3.0c",
        land="white",
#       land="lightgray",
        water="white"
        # shorelines="0.2p,#4A4A4A"
    )
    # Draw national boundaries (automatically clipped to current region)
    fig.plot(data="china_border.gmt", pen="0.1p,#4A4A4A")
    fig.plot(data="bianjing.gmt", pen="0.8p,#4A4A4A")
    fig.plot(data="Loess_Plateau.gmt", pen="1p,#8B4513")

# Save and show the figure
fig.savefig(FIGURES_DIR / "fig1_study_area.png", dpi=300)
fig.show()

