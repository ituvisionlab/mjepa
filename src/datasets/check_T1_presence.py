import pandas as pd

# Paths to the CSV files
ppmi_csv_path = "ppmi_all_nii.csv"
oasis3_csv_path = "oasis3_all_nii.csv"
adni_csv_path = "adni_all_nii.csv"

def check_t1_contrast(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by subject_id and check if any row has contrast 'T1'
    subject_has_t1 = df.groupby('subject_id')['contrast'].apply(lambda x: (x == 'T1').any())
    
    # Extract subjects without T1 contrast
    subjects_without_t1 = subject_has_t1[~subject_has_t1].index.tolist()
    
    # Print results
    print(f"Total subjects: {len(subject_has_t1)}")
    print(f"Subjects without T1 contrast: {len(subjects_without_t1)}")
    
    if subjects_without_t1:
        print("List of subjects without T1 contrast:")
        print(subjects_without_t1)
    
    return subjects_without_t1

# Check ADNI dataset
print("\nChecking ADNI dataset...")
adni_subjects_without_t1 = check_t1_contrast(adni_csv_path)

# Check PPMI dataset
#print("Checking PPMI dataset...")
#ppmi_subjects_without_t1 = check_t1_contrast(ppmi_csv_path)

# Check OASIS3 dataset
#print("\nChecking OASIS3 dataset...")
#oasis3_subjects_without_t1 = check_t1_contrast(oasis3_csv_path)
