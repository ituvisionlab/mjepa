import os
import pandas as pd

nifti_root = "/gpfs/data/prostatelab/NIFTI"
metadata_file = "/gpfs/data/prostatelab/NIFTI_csv/Prostate_training_6May2025.csv"

# Read metadata
df = pd.read_csv(metadata_file, dtype=str)  # Read everything as string for safe comparison

# Get all folder names under NIFTI/
existing_folders = set(os.listdir(nifti_root))

# Check how many AccNum entries match folders
accnum_matches = df["AccNum"].apply(lambda x: x in existing_folders)
patientid_matches = df["PatientID"].apply(lambda x: x in existing_folders)

print(f"Matches by AccNum: {accnum_matches.sum()} / {len(df)}")
print(f"Matches by PatientID: {patientid_matches.sum()} / {len(df)}")

# View a few mismatches for debugging
print("First few missing AccNum values:")
print(df[~accnum_matches]["AccNum"].head())

print("First few missing PatientID values:")
print(df[~patientid_matches]["PatientID"].head())
