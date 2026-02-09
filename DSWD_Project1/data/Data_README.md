# Data Description

This directory contains input datasets for the DSWD-RF analysis.

## Files

| File | Rows | Description | Used by |
|------|------|-------------|---------|
| `sites_72_environmental.csv` | 72 | Site-level environmental variables | Full-dataset model |
| `sites_24_metadata.csv` | 24 | Site metadata for subset | Subset models |
| `sites_72_coordinates.csv` | 72 | Site coordinates (lon, lat) | Mapping |
| `bam.txt` | 256 | Colormap file (blue-amber-magenta) | Visualization |
| `vik.txt` | 256 | Colormap file (diverging) | Visualization |
| `gmt/` | - | GMT boundary files for mapping | Mapping |

## Column Descriptions

### sites_72_environmental.csv

| Column | Description | Unit |
|--------|-------------|------|
| site_id | Site identifier | - |
| DSWD_mm | Deep Soil Water Deficit | mm |
| MAP_mm | Mean Annual Precipitation | mm |
| PET_mm | Potential Evapotranspiration | mm |
| clay_pct | Clay content | % |
| sand_pct | Sand content | % |
| silt_pct | Silt content | % |
| stand_age_yr | Stand age | years |
| species_code | Tree species code | - |
| lon | Longitude | ° |
| lat | Latitude | ° |

### sites_24_metadata.csv

| Column | Description | Unit |
|--------|-------------|------|
| site_id | Site identifier | - |
| species_code | Tree species code | - |
| lon | Longitude | ° |
| lat | Latitude | ° |
| stand_age_yr | Stand age | years |
| MAP_original_mm | Original precipitation measurement | mm |
| MAP_gridded_mm | Gridded precipitation product | mm |
| planting_density_ha | Planting density | plants/ha |
| root_deepening_rate_m_yr | Root deepening rate | m/yr |
| max_rooting_depth_m | Maximum rooting depth | m |
| PET_mm | Potential Evapotranspiration | mm |
| clay_pct | Clay content | % |
| sand_pct | Sand content | % |
| silt_pct | Silt content | % |
| root_C_shallow_Mg_ha | Root C input (shallow soil) | Mg/ha |
| root_C_deep_Mg_ha | Root C input (deep soil) | Mg/ha |
| SOC_deep_pct | Deep soil organic carbon | % |

## Species Codes

| Code | Species |
|------|---------|
| 100 | Robinia pseudoacacia (Black Locust) |
| 101 | Caragana korshinskii |
| 102 | Pinus tabuliformis (Chinese Pine) |
| 103 | Populus spp. (Poplar) |
| 104 | Platycladus orientalis (Oriental Arborvitae) |
| 105 | Hippophae rhamnoides (Sea Buckthorn) |
| ... | ... |

## Data Availability

Raw soil water content profile data and root trait measurements are available upon reasonable request to the corresponding author. Pre-computed model outputs (predictions and feature importance) are provided in the `results/` directory for result verification.
