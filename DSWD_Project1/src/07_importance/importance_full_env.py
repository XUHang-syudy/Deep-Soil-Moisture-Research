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
rawdata = pd.read_csv(DATA_DIR / "sites_72_environmental.csv", encoding='gbk')
rawdata2 = pd.read_csv(DATA_DIR / "profiles_72_soilwater.csv", encoding='gbk')
rawdata2_row = np.size(rawdata2, 0)

err = np.empty((0, 11))
# Iterate through 72 monitoring sites
for k in range(1, 73):
    site_sum = np.empty((0, 11))
    site = np.empty((0, 4))
    for i in range(rawdata2_row):
        if rawdata2.iloc[i, 0] == k:
            site = np.vstack((site, rawdata2.iloc[i, :]))
    df = pd.DataFrame(site)
    for j in range(df.shape[0]):
        if df.iloc[j, 1] > 3.0:
            # Calculate cumulative sum for the specific vertical profile
            df['sum1'] = df.iloc[15:j+1, :].sum(axis=0)
            site_sum_hang = pd.DataFrame(np.empty((1, 11)))
            # Feature engineering: DSWD calculation and depth
            site_sum_hang.iloc[0, 0] = (df.iloc[2, 4] - df.iloc[3, 4]) * 200
            site_sum_hang.iloc[0, 1] = df.iloc[j, 1] + 0.1 # depth_m
            site_sum_hang.iloc[0, 2:11] = rawdata.iloc[k - 1, 2:11]
            site_sum = np.vstack((site_sum, site_sum_hang))
    err = np.vstack((err, site_sum))

err_df = pd.DataFrame(err)

# --- Variable Renaming (per mapping table) ---
new_feature_names = {
    1: 'depth_m',
    2: 'MAP_mm',
    3: 'PET_mm',
    4: 'clay_pct',
    5: 'sand_pct',
    6: 'silt_pct',
    7: 'stand_age_yr',
    8: 'species_code'
}

# Initialize results container
results = pd.DataFrame()

# Iterate through depth gradients from 3.2m to 25.4m
for depth in np.arange(3.2, 25.4, 0.2):
    # Filter data within the current depth range
    # Use iloc[:, 1] to filter by depth_m column
    depth_data = err_df[(err_df.iloc[:, 1] >= 3.2) & (err_df.iloc[:, 1] <= depth)]
    
    # Ensure statistical robustness: n >= 20
    if len(depth_data) >= 20:
        X = depth_data.iloc[:, 1:9].copy()
        Y = -depth_data.iloc[:, 0]

        # Rename to scientific columns
        X.rename(columns=new_feature_names, inplace=True)

        importances_list = []

        # 100 random split iterations
        for _ in range(100):
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.3, random_state=np.random.randint(0, 10000)
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

        # Calculate mean and std
        importances_mean = np.mean(importances_list, axis=0)
        importances_std = np.std(importances_list, axis=0)

        # Construct result row
        result_row = pd.DataFrame([importances_mean], columns=[f'{col}_mean' for col in X.columns])
        result_row[[f'{col}_std' for col in X.columns]] = [importances_std]
        result_row['depth_m'] = depth  

        # Merge results
        results = pd.concat([results, result_row], ignore_index=True)

# Export results
results.to_excel(RESULTS_DIR / "feature_importance" / "RF_importance_full_background.xlsx", index=False)
print("✅ Analysis complete. Results exported with new scientific naming.")