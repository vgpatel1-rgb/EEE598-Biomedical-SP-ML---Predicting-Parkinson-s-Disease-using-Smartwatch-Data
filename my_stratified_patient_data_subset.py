import os
import json
import pandas as pd

# ================= CONFIG =================
patients_path = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\pads-parkinsons-disease-smartwatch-dataset-1.0.0\pads-parkinsons-disease-smartwatch-dataset-1.0.0\patients"
output_csv_path_original = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\Parkinsons_CSV_Output\original_patient_data_stratified.csv"
output_csv_path_modified = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\Parkinsons_CSV_Output\modified_patient_data_stratified.csv"

# Stratified patient IDs
stratified_patient_ids = [
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

# Fields from original file
base_fields = [
    "resource_type",
    "id",
    "study_id",
    "condition",
    "disease_comment",
    "age_at_diagnosis",
    "age",
    "height",
    "weight",
    "gender",
    "handedness",
    "appearance_in_kinship",
    "appearance_in_first_grade_kinship",
    "effect_of_alcohol_on_tremor"
]

# ================= HELPERS =================
def assign_label(condition):
    if condition == "Healthy":
        return 0
    elif condition == "Parkinson's":
        return 1
    else:
        return 2

def encode_gender(gender):
    gender = str(gender).lower()
    if gender == "male": return 1
    if gender == "female": return 0
    return None

def encode_handedness(hand):
    hand = str(hand).lower()
    if hand == "right": return 1
    if hand == "left": return 0
    return None

def encode_appearance_in_kinship(kin):
    kin = str(kin).upper()
    if kin == "TRUE": return 1
    if kin == "FALSE": return 0
    return None

# ================= LOAD AND BUILD ORIGINAL DATA =================
all_patients_data = []

print("=== Creating Original CSV for Stratified Patients ===")
for file_name in os.listdir(patients_path):
    if file_name.endswith(".json") and file_name.startswith("patient_"):
        patient_id = file_name.split("_")[1].split(".")[0]  # extract ID
        if patient_id not in stratified_patient_ids:
            continue  # skip patients not in the stratified list
        file_path = os.path.join(patients_path, file_name)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            row = {field: data.get(field, None) for field in base_fields}
            row["label"] = assign_label(row["condition"])
            all_patients_data.append(row)
        except Exception as e:
            print(f"Warning: Could not process file {file_name} — {e}")

df_original = pd.DataFrame(all_patients_data)
df_original.to_csv(output_csv_path_original, index=False)
print(f"✅ Original stratified patient data saved to: {output_csv_path_original}")

# ================= MODIFY DATA AS REQUESTED =================
print("=== Creating Modified CSV with Encoded Features ===")
df_modified = df_original.copy()

df_modified["gender"] = df_modified["gender"].apply(encode_gender)
df_modified["handedness"] = df_modified["handedness"].apply(encode_handedness)
df_modified["diagnosis_type"] = df_modified["condition"].apply(assign_label)
df_modified["appearance_in_kinship"] = df_modified["appearance_in_kinship"].apply(encode_appearance_in_kinship)

# Remove unwanted columns
df_modified.drop(columns=["effect_of_alcohol_on_tremor", "appearance_in_first_grade_kinship"], inplace=True, errors='ignore')
df_modified.drop(df_modified.columns[[0, 2, 4]], axis=1, inplace=True)  # optional if needed

df_modified.to_csv(output_csv_path_modified, index=False)
print(f"✅ Modified stratified patient data saved to: {output_csv_path_modified}")

csv_path = r"C:\Users\Veraj\Desktop\Veraj\ASU\EEE598 Biomedical Signal Processing and Machine Learning\Project\Parkinsons_CSV_Output\modified_patient_data_stratified.csv"
df = pd.read_csv(csv_path)

# Total patients
total_patients = len(df)

# Gender counts
men_count = df['gender'].sum()  # since male=1, female=0
women_count = total_patients - men_count

# Handedness counts
right_handed_count = df['handedness'].sum()  # right=1, left=0
left_handed_count = total_patients - right_handed_count

# Parkinson's disease count
pd_count = (df['diagnosis_type'] == 1).sum()
no_pd_count = (df['diagnosis_type'] == 0).sum()

print(f"Total stratified patients: {total_patients}")
print(f"Men: {men_count}")
print(f"Women: {women_count}")
print(f"Right-handed: {right_handed_count}")
print(f"Left-handed: {left_handed_count}")
print(f"Parkinson's patients: {pd_count}")
print(f"Healthy patients: {no_pd_count}")
