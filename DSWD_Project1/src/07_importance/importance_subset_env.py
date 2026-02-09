import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
(RESULTS_DIR / "feature_importance").mkdir(parents=True, exist_ok=True)

# === Step 1: Data Loading and Preprocessing ===
rawdata_path = DATA_DIR / "sites_24_metadata.csv"
rawdata2_path = DATA_DIR / "profiles_24_soilwater_root.csv"
rawdata = pd.read_csv(rawdata_path, encoding='gbk')
rawdata2 = pd.read_csv(rawdata2_path, encoding='gbk')

rawdata2_row = np.size(rawdata2, 0)
err = np.empty((0, 20)) 

for k in range(1, 45):
    site_sum = np.empty((0, 20))
    site = np.empty((0, 6))
    for i in range(0, rawdata2_row):
        if rawdata2.iloc[i, 0] == k:
            site = np.vstack((site, rawdata2.iloc[i, :].values))
            
    df = pd.DataFrame(site)
    for j in range(0, df.shape[0]):
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

err_df = pd.DataFrame(err)
x_all_raw = err_df.iloc[:, 1:20]

# --- Variable Renaming (per Subset mapping table) ---
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

x_all_raw.rename(columns=new_feature_names, inplace=True)

# Select subset environmental features (Subset Background Model)
selected_cols = ['depth_m', 'species_code', 'stand_age_yr', 'MAP_gridded_mm', 'PET_mm', 'clay_pct', 'sand_pct', 'silt_pct']
x_all = x_all_raw[selected_cols]

# Initialize results container
results = pd.DataFrame()

# Analyze feature importance across a depth gradient (3.2m to 25.4m)
for depth in np.arange(3.2, 25.4, 0.2):
    # Filter data subset
    mask = (err_df.iloc[:, 1] >= 3.2) & (err_df.iloc[:, 1] <= depth)
    x_sub = x_all[mask]
    y_sub = -err_df[mask].iloc[:, 0] # DSWD inverted
    
    if len(x_sub) >= 20:
        importances_list = []

        # 100 iterations for uncertainty estimation
        for _ in range(100):
            X_train, X_test, Y_train, Y_test = train_test_split(
                x_sub, y_sub, test_size=0.3, random_state=np.random.randint(0, 10000)
            )
            
            rfr = RandomForestRegressor(
                random_state=42,
                n_estimators=60,
                max_features=4,
                min_samples_leaf=1,
                min_samples_split=2,
                max_depth=5
            )
            rfr.fit(X_train, Y_train)
            importances_list.append(rfr.feature_importances_)

        # Compute metrics
        importances_mean = np.mean(importances_list, axis=0)
        importances_std = np.std(importances_list, axis=0)

        # Build result row
        result_row = pd.DataFrame([importances_mean], columns=[f'{col}_mean' for col in x_sub.columns])
        result_row[[f'{col}_std' for col in x_sub.columns]] = [importances_std]
        result_row['depth_m'] = depth 

        results = pd.concat([results, result_row], ignore_index=True)

# Export results
results.to_excel(RESULTS_DIR / "feature_importance" / "RF_importance_subset_background.xlsx", index=False)
print("✅ Subset background analysis complete. Results exported with new scientific naming.")