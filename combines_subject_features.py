# this code combines all of the individual csv files of each subject 
# creates a single csv of all features of all subjects 

import os
import pandas as pd

# ========== CONFIG ==========
features_input_path = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\Parkinsons_CSV_Output\features_subjects"
output_combined_path = os.path.join(features_input_path, "all_stratified_subjects_features.csv")

# ========== MERGE FEATURE FILES ==========
all_feature_files = [f for f in os.listdir(features_input_path) if f.endswith("_features.csv")]

combined_df = pd.DataFrame()

for file in all_feature_files:
    file_path = os.path.join(features_input_path, file)
    try:
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        print(f" Loaded: {file}")
    except Exception as e:
        print(f" Failed to load {file}: {e}")

# ========== SAVE COMBINED FILE ==========
combined_df.to_csv(output_combined_path, index=False)
print("\n Successfully combined all features!")
print(f" Saved to: {output_combined_path}")
print(f"  Total subjects in combined file: {combined_df.shape[0]}")
print(f" Total features per subject: {combined_df.shape[1] - 1} (excluding subject_id column)")