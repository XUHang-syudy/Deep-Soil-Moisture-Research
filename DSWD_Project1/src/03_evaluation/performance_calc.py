import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
(RESULTS_DIR / "model_predictions").mkdir(parents=True, exist_ok=True)
# ====== Model 1: Full-dataset Background Model ======
rawdata = pd.read_csv(DATA_DIR / "sites_72_environmental.csv", encoding='gbk')
rawdata2 = pd.read_csv(DATA_DIR / "profiles_72_soilwater.csv", encoding='gbk')
err = np.empty((0, 11))

for k in range(1, 73):
    site_sum = np.empty((0, 11))
    site = np.empty((0, 4))
    for i in range(rawdata2.shape[0]):
        if rawdata2.iloc[i, 0] == k:
            site = np.vstack((site, rawdata2.iloc[i, :]))
    
    df = pd.DataFrame(site)
    for j in range(df.shape[0]):
        if df.iloc[j, 1] > 3.0:
            df['sum1'] = df.iloc[15:j+1, :].sum(axis=0)
            site_sum_hang = pd.DataFrame(np.empty((1, 11)))
            site_sum_hang.iloc[0, 0] = (df.iloc[2, 4] - df.iloc[3, 4]) * 200 # DSWD_mm
            site_sum_hang.iloc[0, 1] = df.iloc[j, 1] + 0.1 # depth_m
            site_sum_hang.iloc[0, 2:11] = rawdata.iloc[k - 1, 2:11]
            site_sum = np.vstack((site_sum, site_sum_hang))
    err = np.vstack((err, site_sum))

err1 = pd.DataFrame(err)

rename_map1 = {
    1: 'depth_m', 2: 'MAP_mm', 3: 'PET_mm', 4: 'clay_pct',
    5: 'sand_pct', 6: 'silt_pct', 7: 'stand_age_yr', 8: 'species_code',
    9: 'lon', 10:'lat'
}
x1 = err1.iloc[:, 1:11].rename(columns=rename_map1)
y1 = err1.iloc[:, 0]


sel_cols1 = ['depth_m', 'species_code', 'stand_age_yr', 'MAP_mm', 'PET_mm', 'clay_pct', 'sand_pct', 'silt_pct']
x1 = x1[sel_cols1]

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(x1, y1, test_size=0.3, random_state=42)
rfr1 = RandomForestRegressor(random_state=42, n_estimators=60, max_features=4, min_samples_leaf=1, min_samples_split=2, max_depth=5)
rfr1.fit(X_train1, Y_train1)

# ====== Model 2 & 3: Subset Models ======
rawdata_sub = pd.read_csv(DATA_DIR / "sites_24_metadata.csv", encoding='gbk')
rawdata2_sub = pd.read_csv(DATA_DIR / "profiles_24_soilwater_root.csv", encoding='gbk')
err_sub = np.empty((0, 20))

for k in range(1, 45):
    site_sum = np.empty((0, 20))
    site = np.empty((0, 6))
    for i in range(rawdata2_sub.shape[0]):
        if rawdata2_sub.iloc[i, 0] == k:
            site = np.vstack((site, rawdata2_sub.iloc[i, :].values))
            
    df = pd.DataFrame(site)
    for j in range(df.shape[0]):
        if df.iloc[j, 1] > 3.0:
            df['sum1'] = df.iloc[15:j+1, :].sum(axis=0)
            site_sum_hang = pd.DataFrame(np.empty((1, 20)))
            site_sum_hang.iloc[0, 0] = (df.iloc[2, 6] - df.iloc[3, 6]) * 200
            site_sum_hang.iloc[0, 1] = df.iloc[j, 1] + 0.1 # depth_m
            site_sum_hang.iloc[0, 2] = df.iloc[4, 6] # RDWD_kg_m3
            site_sum_hang.iloc[0, 3] = df.iloc[5, 6] # FRLD_m_m3
            
            row_indices = np.where(rawdata_sub.iloc[:, 0] == k)[0]
            site_sum_hang.iloc[0, 4:20] = rawdata_sub.iloc[row_indices, 1:17].values
            site_sum = np.vstack((site_sum, site_sum_hang))
    err_sub = np.vstack((err_sub, site_sum))

err_sub_df = pd.DataFrame(err_sub)

# Mapping Table 2 (includes root traits and detailed parameters)
rename_map_sub = {
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

# --- Model 2: Subset Background (Environmental factors only) ---
x2 = err_sub_df.iloc[:, 1:20].rename(columns=rename_map_sub)
y2 = err_sub_df.iloc[:, 0]
sel_cols2 = ['depth_m', 'species_code', 'stand_age_yr', 'MAP_gridded_mm', 'PET_mm', 'clay_pct', 'sand_pct', 'silt_pct']
# Note: Model 2 uses gridded precipitation (MAP_gridded_mm)
x2 = x2[sel_cols2]

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=42)
rfr2 = RandomForestRegressor(random_state=42, n_estimators=60, max_features=4, min_samples_leaf=1, min_samples_split=2, max_depth=5)
rfr2.fit(X_train2, Y_train2)

# --- Model 3: Subset Full-variable (Environmental + Root Traits) ---
x3 = err_sub_df.iloc[:, 1:20].rename(columns=rename_map_sub)
y3 = err_sub_df.iloc[:, 0]
sel_cols3 = ['depth_m', 'RDWD_kg_m3', 'FRLD_m_m3', 'species_code', 'lat', 'lon', 
             'stand_age_yr', 'MAP_gridded_mm', 'PET_mm', 'clay_pct', 'sand_pct', 'silt_pct']
x3 = x3[sel_cols3]

X_train3, X_test3, Y_train3, Y_test3 = train_test_split(x3, y3, test_size=0.3, random_state=42)
rfr3 = RandomForestRegressor(random_state=42, n_estimators=60, max_features=4, min_samples_leaf=1, min_samples_split=2, max_depth=5)
rfr3.fit(X_train3, Y_train3)

# ====== Export Function ======
def export_predictions(depth, observed, predicted, filepath):
    depth = pd.Series(depth).reset_index(drop=True)
    observed = pd.Series(observed).reset_index(drop=True)
    predicted = pd.Series(predicted, name='Predicted_DSWD_mm')

    depth_repeat = pd.concat([depth, depth], ignore_index=True)
    type_col = ['Observed DSWD'] * len(depth) + ['Predicted DSWD'] * len(depth)
    swsd_values = pd.concat([observed, predicted], ignore_index=True)

    export_df = pd.DataFrame({
        'depth_m': depth_repeat,
        'DSWD_value_mm': swsd_values,
        'data_type': type_col
    })

    export_df.to_excel(filepath, index=False)
    print(f"✅ Exported: {filepath}")

# --- Execute Predictions and Export ---
export_predictions(X_test1['depth_m'], Y_test1, rfr1.predict(X_test1), RESULTS_DIR / "model_predictions" / "predictions_full_background.xlsx")
export_predictions(X_test2['depth_m'], Y_test2, rfr2.predict(X_test2), RESULTS_DIR / "model_predictions" / "predictions_subset_background.xlsx")
export_predictions(X_test3['depth_m'], Y_test3, rfr3.predict(X_test3), RESULTS_DIR / "model_predictions" / "predictions_subset_full.xlsx")