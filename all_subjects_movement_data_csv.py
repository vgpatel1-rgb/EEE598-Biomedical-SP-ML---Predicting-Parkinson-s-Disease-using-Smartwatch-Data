# this creats the .csv with all of the movement data of all of the stratified subjects 



import os
import numpy as np
import pandas as pd


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

# ================= CONFIG =================


raw_data_path = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\pads-parkinsons-disease-smartwatch-dataset-1.0.0\pads-parkinsons-disease-smartwatch-dataset-1.0.0\movement\timeseries"
output_folder = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\Parkinsons_CSV_Output\movement_subjects"

os.makedirs(output_folder, exist_ok=True)

movement_order = [
    "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
    "PointFinger", "Relaxed", "RelaxedTask", "StretchHold", "TouchIndex", "TouchNose"
]

hand_aliases = {"LeftWrist": "LeftWrist", "RightWrist": "RightWrist"}
axes = ["X", "Y", "Z"]
sensors = ["Accelerometer", "Gyroscope"]

# ================= PROCESS SUBJECTS =================
for subject_id in stratified_subject_ids:
    print(f"Processing subject {subject_id}...")
    all_data = []
    all_columns = []
    file_counter = 1

    for mov in movement_order:
        for hand_file, hand_name in hand_aliases.items():
            filename = f"{subject_id}_{mov}_{hand_file}.txt"
            filepath = os.path.join(raw_data_path, filename)

            cols = [f"TimeSeries_{file_counter}"]
            for sensor in sensors:
                for axis in axes:
                    cols.append(f"{mov}_{hand_name}_{sensor}_{axis}")
            file_counter += 1
            all_columns.extend(cols)

            if os.path.exists(filepath):
                try:
                    data = np.loadtxt(filepath, delimiter=",")
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    if data.shape[1] < 7:
                        padded = np.full((data.shape[0], 7), np.nan)
                        padded[:, :data.shape[1]] = data
                        data = padded
                    elif data.shape[1] > 7:
                        data = data[:, :7]
                except Exception as e:
                    print(f"Warning: couldn't read file {filename} — using NaN ({e})")
                    data = np.full((1000, 7), np.nan)
            else:
                print(f"Warning: file not found {filename} — using NaN")
                data = np.full((1000, 7), np.nan)

            all_data.append(data)

    # Truncate to shortest file
    min_rows = min(d.shape[0] for d in all_data)
    truncated_data = [d[:min_rows, :] for d in all_data]

    # Horizontal concat
    combined_data = np.hstack(truncated_data)

    # Create DataFrame
    df_subject = pd.DataFrame(combined_data, columns=all_columns)

    # Remove TimeSeries columns
    time_series_cols = [col for col in df_subject.columns if col.startswith("TimeSeries")]
    df_subject = df_subject.drop(columns=time_series_cols)

    # Remove first 50 and last 10 rows
    df_subject = df_subject.iloc[50:-10]

    # Save subject CSV
    output_csv_path = os.path.join(output_folder, f"subject_{subject_id}_movement.csv")
    df_subject.to_csv(output_csv_path, index=False)
    print(f"Saved cleaned CSV for subject {subject_id}, shape: {df_subject.shape}")

print("✅ All stratified subjects processed and saved.")

