#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --job-name=unzip2
#SBATCH --output=unzip_%j.out
#SBATCH --error=unzip_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# --- Define paths ---
ZIP_FILE="/gpfs/data/sodicksonlab/gozde/SCAN/SCAN_MRI_Download2.zip"
OUTPUT_DIR="/gpfs/data/sodicksonlab/gozde/SCAN"
cd "$OUTPUT_DIR"


# Run the unzip command
#unzip "$ZIP_FILE" -d "$OUTPUT_DIR"
unzip "$ZIP_FILE"

echo "Unzipping complete."
