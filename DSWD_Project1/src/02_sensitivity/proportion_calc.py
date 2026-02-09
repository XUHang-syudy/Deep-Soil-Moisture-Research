import os
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import time

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
(RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

# Set global font parameters to Arial
plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 20

# Load datasets
rawdata = pd.read_csv(DATA_DIR / "sites_72_environmental.csv", encoding='gbk')
rawdata2 = pd.read_csv(DATA_DIR / "profiles_72_soilwater.csv", encoding='gbk')

rawdata2_row = np.size(rawdata2, 0)  # Calculate number of rows in rawdata2
rawdata2_col = np.size(rawdata2, 1)  # Calculate number of columns in rawdata2
err = np.empty((0, 11)) 

# Iterate through monitoring sites (ID 1 to 72)
for k in range(1, 73):
    site_sum = np.empty((0, 11))
    site = np.empty((0, 4))
    for i in range(0, rawdata2_row):
        if rawdata2.iloc[i, 0] == k:
            site = np.vstack((site, rawdata2.iloc[i, :]))
    site_row = np.size(site, 0) 
    site_col = np.size(site, 1)
    df = pd.DataFrame(site)  # Convert list to DataFrame
    for j in range(0, site_row):
        if df.iloc[j, 1] > 3.0:
            # Calculate cumulative sum for specified indices
            df['sum1'] = df.iloc[15:j+1, :].sum(axis=0)
            site_sum_hang = np.empty((1, 11))
            depth = df.iloc[j, 1]  # Extract depth
            site_sum_hang = pd.DataFrame(site_sum_hang)
            # Calculate cumulative value based on site-specific parameters
            site_sum_hang.iloc[0, 0] = (df.iloc[2, 4] - df.iloc[3, 4]) * 200 
            site_sum_hang.iloc[0, 1] = df.iloc[j, 1] + 0.1  # Add depth
            site_sum_hang.iloc[0, 2:11] = rawdata.iloc[k-1, 2:11]  # Append auxiliary variables
            site_sum = np.vstack((site_sum, site_sum_hang))
    err = np.vstack((err, site_sum))

# Post-processing integrated data
err = pd.DataFrame(err)
C = err.drop_duplicates()
x = err.iloc[:, 1:9]  # Input features
y = -err.iloc[:, 0]  # Target variable (DSWD)

# List to store iteration results
results = []

# Outer loop: vary the test_size from 0.01 to 0.99
for test_size in np.arange(0.01, 0.995, 0.01):
    temp_mse_train = []
    temp_rmse_train = []
    temp_r2_train = []
    temp_mse_test = []
    temp_rmse_test = []
    temp_r2_test = []
    temp_run_time = []

    # Inner loop: repeat 100 times for each proportion to ensure statistical robustness
    for repeat in range(100):
        start_time = time.time()

        # Split dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        # Initialize and fit Random Forest model with optimized hyperparameters
        model = RandomForestRegressor(random_state=42, n_estimators=60, max_features=4, 
                                      min_samples_leaf=1, min_samples_split=2, max_depth=5)
        model.fit(x_train, y_train)

        # Predict on training and testing sets
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Calculate metrics for training set
        mse_train = mean_squared_error(y_train, y_train_pred)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(y_train, y_train_pred)

        # Calculate metrics for testing set
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_test_pred)

        end_time = time.time()
        run_time = end_time - start_time

        # Store temporary results for this repeat
        temp_mse_train.append(mse_train)
        temp_rmse_train.append(rmse_train)
        temp_r2_train.append(r2_train)
        temp_mse_test.append(mse_test)
        temp_rmse_test.append(rmse_test)
        temp_r2_test.append(r2_test)
        temp_run_time.append(run_time)

    # Calculate mean and standard deviation for the current test_size
    results.append([
        test_size,
        np.mean(temp_mse_train), np.std(temp_mse_train),
        np.mean(temp_rmse_train), np.std(temp_rmse_train),
        np.mean(temp_r2_train), np.std(temp_r2_train),
        np.mean(temp_mse_test), np.std(temp_mse_test),
        np.mean(temp_rmse_test), np.std(temp_rmse_test),
        np.mean(temp_r2_test), np.std(temp_r2_test),
        np.mean(temp_run_time), np.std(temp_run_time),
        len(y_train), len(y_test)
    ])

# Define columns for the results DataFrame
columns = [
    'test_size',
    'mse_train_mean', 'mse_train_std',
    'rmse_train_mean', 'rmse_train_std',
    'r2_train_mean', 'r2_train_std',
    'mse_test_mean', 'mse_test_std',
    'rmse_test_mean', 'rmse_test_std',
    'r2_test_mean', 'r2_test_std',
    'run_time_mean', 'run_time_std',
    'y_train_size', 'y_test_size'
]

results_df = pd.DataFrame(results, columns=columns)

# Save the final performance metrics to an Excel file
results_df.to_excel(RESULTS_DIR / "tables" / "cv_performance_100runs.xlsx", index=False)
