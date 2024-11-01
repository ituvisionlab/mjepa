import torch.utils.tensorboard as tensorboard
import csv
import os

def change_absolute_paths(input_csv):
    with open(input_csv, mode='r') as infile:
        reader = csv.reader(infile)
        
        log_path = "./logs"
        
        date_time = input_csv.split(".")[0]
        
        log_dir = os.path.join(log_path, date_time)
        
        os.makedirs(log_dir, exist_ok=True)
        
        writer = tensorboard.SummaryWriter(log_dir)
        
        # Write the header row
        header = next(reader)
        
        itr_counter = 0
        # Filter out rows with "Spatially_Normalized" in the path
        for row in reader:
            
            if row[0] == "epoch":
                continue
            else:
                writer.add_scalar('loss', float(row[2]), itr_counter)
                writer.add_scalar('loss-jepa', float(row[3]), itr_counter)
                writer.add_scalar('reg-loss', float(row[4]), itr_counter)
                writer.add_scalar('enc-grad-norm', float(row[5]), itr_counter)
                writer.add_scalar('pred-grad-norm', float(row[6]), itr_counter)
                writer.add_scalar('gpu-time(ms)', float(row[7]), itr_counter)
                writer.add_scalar('wall-time(ms)', float(row[8]), itr_counter)
                writer.flush()
                
                itr_counter += 1

                
            # # Check if "Spatially_Normalized" is not present in the file path
            # row[2] = row[2].replace("/media/yusuf/backup", "/media/disk2")
            # writer.writerow(row)
    
    # print(f"tensorboard log created at: {log_dir}")

# Example usage
file_list = ["evals/jepa_r0-Oct27train.csv", "evals/jepa_r1-Oct27train.csv", "evals/jepa_r2-Oct27train.csv", "evals/jepa_r3-Oct27train.csv"]

for file in file_list:
    change_absolute_paths(file)
