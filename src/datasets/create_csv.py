import csv
import os

# Step 1: Read the labels from the first CSV file and map them to integers
label_to_int = {}
youtube_id_to_label = {}

with open('test_files.csv', 'r') as f1:
    csv_reader = csv.reader(f1, delimiter=',')
    next(csv_reader)  # Skip the first description row
    for row in csv_reader:
        label = row[0]  # First column is the label
        youtube_id = row[1]  # Second column is the YouTube ID
        if label not in label_to_int:
            label_to_int[label] = len(label_to_int)  # Assign each label a unique integer
        youtube_id_to_label[youtube_id] = label_to_int[label]  # Map YouTube ID to the label

# Step 2: Read the MP4 filenames from the second CSV file
with open('test_file_list.csv', 'r') as f2, open('video_test.csv', 'w', newline='') as out_csv:
    csv_writer = csv.writer(out_csv, delimiter=' ')
    # Write rows in the format: /absolute_file_path.mp4 $integer_class_label
    for row in f2:
        mp4_filename = row.strip()
        youtube_id = mp4_filename.split('_')[0]  # Extract the YouTube ID part
        class_label = youtube_id_to_label.get(youtube_id)
        if class_label is not None:
            #abs_file_path = os.path.abspath(mp4_filename)  # Get the absolute file path
            abs_file_path = '/media/yusuf/backup/kinetics-dataset/k400/test/'+mp4_filename
            csv_writer.writerow([abs_file_path, f"{class_label}"])
