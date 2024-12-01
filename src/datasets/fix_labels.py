import csv
import pandas as pd

def change_absolute_paths(input_csv, output_csv):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=" ")
        writer = csv.writer(outfile, delimiter=" ")
        
        with open("correct_label_dict.txt") as f:
            label_lines = f.readlines()
            
            label_lines = [l.strip("\n,'") for l in label_lines]
            
            label_dict = {}
            
            for line in label_lines:
                label_id, val = line.split(" ")[0], " ".join(line.split(" ")[1:])
                label_dict[val] = label_id
                
            original_labels = pd.read_csv("original_labels.csv")
            
        
        # Write the header row
        # header = next(reader)
        # writer.writerow(header)
        
        for row in reader:
            new_name = original_labels.loc[original_labels["id"] == int(row[1])]["name"].values[0]
            row[1] = str(label_dict[new_name])
            writer.writerow(row)
    
    print(f"CSV file with changed paths created: {output_csv}")

# Example usage
dataset_split = "train"

input_csv = f"video_{dataset_split}.csv"  # Path to the combined CSV file
output_csv = f"video_{dataset_split}_correct_labels.csv"  # Path to the filtered CSV file
change_absolute_paths(input_csv, output_csv)
