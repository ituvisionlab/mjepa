import pickle
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Read a pickle file using a specified job ID.")
parser.add_argument("job_id", type=str, help="The job ID to locate the pickle file.")
args = parser.parse_args()

# Construct the file path using the provided job ID
file_path = f"/gpfs/home/unalg01/jepa/job_{args.job_id}/{args.job_id}_2_result.pkl"
#file_path = f"/gpfs/home/unalg01/jepa/job_{args.job_id}/{args.job_id}_submitted.pkl"
# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load the pickle file
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    # Print the loaded data
    print(data)
    
    # Safely print object attributes
    try:
        print(vars(data))
    except TypeError:
        print("The data object does not have a __dict__ attribute.")
    
    # Access the trainer function if present
    try:
        trainer = data.function
        print(vars(trainer))  # Print the Trainer object’s stored attributes
    except AttributeError:
        print("The data object does not have a 'function' attribute.")

