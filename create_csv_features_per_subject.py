# this code creates a new csv file for each subject
# the information is just the extracted features

import os
import pandas as pd
import numpy as np

# ================= CONFIG =================
stratified_subject_ids = [
    "001", "002", "003", "004", "006", "007", "011", "014", "016", "017", "018", "021", "023",
    "024", "027", "032", "034", "035", "036", "041", "042", "045", "046", "049", "052", "054",
    "056", "058", "059", "062", "063", "064", "065", "069", "070", "071", "072", "075", "077",
    "078", "079", "080", "081", "084", "085", "086", "088", "089", "091", "097", "098", "100",
    "101", "102", "104", "107", "108", "109", "111", "112", "114", "118", "119", "123", "126",
    "127", "132", "134", "135", "137", "141", "142", "143", "144", "145", "147", "148", "153",
    "156", "157", "160", "161", "162", "164", "165", "169", "172", "180", "181", "183", "185",
    "188", "191", "193", "195", "196", "199", "200", "201", "203", "204", "205", "207", "209",
    "211", "212", "213", "216", "219", "223", "224", "225", "227", "228", "229", "230", "231",
    "232", "237", "239", "241", "244", "245", "246", "250", "253", "254", "256", "259", "263",
    "270", "272", "275", "276", "278", "280", "282", "283", "284", "285", "288", "289", "292",
    "293", "295", "298", "304", "305", "310", "312", "314", "315", "318", "319", "320", "326",
    "330", "334", "335", "340", "341", "350", "351", "353", "360", "361", "363", "367", "370",
    "373", "377", "379", "385", "388", "389", "392", "393", "396", "398", "399", "402", "406",
    "416", "417", "419", "421", "423", "427", "430", "434", "435", "443", "444", "445", "446",
    "449", "450", "453", "457", "462", "464", "466", "469"
]

movement_csv_path = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\Parkinsons_CSV_Output\movement_subjects"
features_output_path = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\Parkinsons_CSV_Output\features_subjects"

# Create output directory if it doesn't exist
os.makedirs(features_output_path, exist_ok=True)

# ================= MOVEMENT SETTINGS =================
movements = [
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex", "TouchNose"
]
hands = ["LeftWrist", "RightWrist"]
sensors = ["Accelerometer", "Gyroscope"]
axes = ["X", "Y", "Z"]

def compute_features(df, subj_id):
    features = {"subject_id": subj_id}
    sma_temp = {}
    for mov in movements:
        for hand in hands:
            for sensor in sensors:
                for axis in axes:
                    col_name_in_csv = f"{mov}_{hand}_{sensor}_{axis}"
                    feature_prefix = f"{mov}_{hand}_{sensor}_{axis}"
                    
                    # ================= SMA (Signal Magnitude Area) =================
# For each movement, hand, and sensor, sum |x| + |y| + |z| properly
                    cols = [f"{mov}_{hand}_{sensor}_{ax}" for ax in axes]
                    if all(col in df.columns for col in cols):
                        sma_value = np.sum(np.abs(df[cols[0]].fillna(0)) +
                                           np.abs(df[cols[1]].fillna(0)) +
                                           np.abs(df[cols[2]].fillna(0)))
                    else:
                        sma_value = np.nan

    # Store SMA feature
                    features[f"SMA_{mov}_{hand}_{sensor}"] = sma_value

                    if col_name_in_csv in df.columns:
                        data = df[col_name_in_csv]
                        features[f"mean_{feature_prefix}"] = data.mean() # mean of each column
                        features[f"rms_{feature_prefix}"] = np.sqrt((data**2).mean()) # rms of each column
                        features[f"std_{feature_prefix}"] = data.std() # std of each column 
                   
                        # Zero Crossing Rate
                        zero_crossings = np.where(np.diff(np.sign(data)))[0]
                        features[f"zcr_{feature_prefix}"] = len(zero_crossings) / len(data)
                       
                        # Accumulate for SMA (sum of |x|,|y|,|z|)
                        sma_key = f"{mov}_{hand}_{sensor}"
                        sma_temp.setdefault(sma_key, 0)
                        sma_temp[sma_key] += np.sum(np.abs(data))
                       
                        # Jerk (first derivative)
                        jerk = np.diff(data)  # assuming dt = 1
                        features[f"jerk_mean_{feature_prefix}"] = jerk.mean() if len(jerk) > 0 else np.nan
                
                        features[f"jerk_rms_{feature_prefix}"] = np.sqrt((jerk**2).mean()) if len(jerk) > 0 else np.nan
                   
                    
                    else:
                        features[f"mean_{feature_prefix}"] = np.nan
                        features[f"rms_{feature_prefix}"] = np.nan
                        features[f"std_{feature_prefix}"] = np.nan
                        features[f"zcr_{feature_prefix}"] = np.nan
                        features[f"jerk_mean_{feature_prefix}"] = np.nan
                        features[f"jerk_rms_{feature_prefix}"] = np.nan
    return features

# ================= PROCESS EACH SUBJECT =================
for subj_id in stratified_subject_ids:
    filename = f"subject_{subj_id}_movement.csv"
    file_path = os.path.join(movement_csv_path, filename)

    if not os.path.exists(file_path):
        print(f"File not found for subject {subj_id}, skipping.")
        continue

    print(f"Processing subject {subj_id}...")
    df = pd.read_csv(file_path)

    # Remove time series or unwanted columns
    time_series_cols = [c for c in df.columns if "TimeSeries" in c]
    df = df.drop(columns=time_series_cols, errors='ignore')

    # Trim first 50 and last 10 rows if needed
    df = df.iloc[50:-10] if len(df) > 60 else df

    # Compute features
    subj_features = compute_features(df, subj_id)

    # Save to CSV
    output_file = os.path.join(features_output_path, f"subject_{subj_id}_features.csv")
    pd.DataFrame([subj_features]).to_csv(output_file, index=False)

    print(f"Features saved to {output_file}\n")

print("Finished processing all subjects!")
