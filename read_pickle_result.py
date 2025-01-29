import pickle
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Read a pickle file using a specified job ID.")
parser.add_argument("job_id", type=str, help="The job ID to locate the pickle file.")
parser.add_argument("number", type=int, help="The number in the pickle output file name: 0, 1, 2....")
args = parser.parse_args()

# Construct the file path using the provided job ID
file_path = f"/gpfs/home/unalg01/jepa/job_{args.job_id}/{args.job_id}_{args.number}_result.pkl"
print(f"Generated file path: {file_path}")

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

# Load the pickle file safely
try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
except (pickle.UnpicklingError, EOFError) as e:
    print(f"Error loading pickle file: {e}")
    exit(1)

# Print the loaded data
print(data)

# Safely print object attributes
if hasattr(data, "__dict__"):
    print(vars(data))
else:
    print("The data object does not have a __dict__ attribute.")

# Access the trainer function if present
if hasattr(data, "function"):
    print(vars(data.function))
else:
    print("The data object does not have a 'function' attribute.")

