
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ================= CONFIG =================
subject_ids = {"healthy": "054", "pd": "056"}
movement_csv_path = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\Parkinsons_CSV_Output\movement_subjects"
movements = [
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex", "TouchNose"
]
sensor_axes = ["X", "Y", "Z"]
sensors = ["Accelerometer", "Gyroscope"]
hand = "RightWrist"

# Colors
colors_healthy = ["blue", "green", "purple"]
colors_pd = ["red", "orange", "pink"]

# Savitzky-Golay parameters
window_length = 51  # must be odd
polyorder = 3

# ================= FUNCTION TO PLOT =================
def plot_movement_comparison(mov):
    plt.figure(figsize=(12, 5))
    
    # ===== Load data =====
    data_healthy = pd.read_csv(f"{movement_csv_path}/subject_{subject_ids['healthy']}_movement.csv")
    data_pd = pd.read_csv(f"{movement_csv_path}/subject_{subject_ids['pd']}_movement.csv")
    
    # Remove TimeSeries columns
    time_cols = [c for c in data_healthy.columns if "TimeSeries" in c]
    data_healthy = data_healthy.drop(columns=time_cols, errors='ignore').iloc[50:-10]
    data_pd = data_pd.drop(columns=time_cols, errors='ignore').iloc[50:-10]

    # ===== Plot accelerometer =====
    plt.subplot(2, 1, 2)
    for i, axis in enumerate(sensor_axes):
        col_name = f"{mov}_{hand}_Accelerometer_{axis}"
        
        # Smooth data using Savitzky-Golay
        if col_name in data_healthy.columns:
            y_h = savgol_filter(data_healthy[col_name].values, window_length, polyorder)
            plt.plot(y_h, color=colors_healthy[i], label=f"Healthy {axis}")
        if col_name in data_pd.columns:
            y_pd = savgol_filter(data_pd[col_name].values, window_length, polyorder)
            plt.plot(y_pd, color=colors_pd[i], label=f"PD {axis}", linestyle=':')
    plt.title(f"{mov} - Accelerometer ({hand})")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.legend()

    # ===== Plot gyroscope =====
    plt.subplot(2, 1, 1)
    for i, axis in enumerate(sensor_axes):
        col_name = f"{mov}_{hand}_Gyroscope_{axis}"
        
        # Smooth data using Savitzky-Golay
        if col_name in data_healthy.columns:
            y_h = savgol_filter(data_healthy[col_name].values, window_length, polyorder)
            plt.plot(y_h, color=colors_healthy[i], label=f"Healthy {axis}")
        if col_name in data_pd.columns:
            y_pd = savgol_filter(data_pd[col_name].values, window_length, polyorder)
            plt.plot(y_pd, color=colors_pd[i], label=f"PD {axis}", linestyle=':')
    plt.title(f"{mov} - Gyroscope ({hand})")
    plt.xlabel("Time")
    plt.ylabel("Angular velocity")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ================= PLOT ALL MOVEMENTS =================
for mov in movements:
    plot_movement_comparison(mov)
