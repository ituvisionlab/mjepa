#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --job-name=downscan
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/download_scan_%j.log

# Optional: load modules if curl is not available by default
# module load curl

# Target directory
TARGET_DIR="/gpfs/data/sodicksonlab/gozde/SCAN"

# File name for output
#OUTPUT_FILE="SCAN_UDS_NACC69.csv"
OUTPUT_FILE="SCAN_MRI_Download2.zip"

# Create target directory if not exists
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"


# Actual download command with resume support
curl -C - "https://ida.loni.usc.edu/download/files/ida2/9181c9d8-67dd-4cf1-ba18-a7d96390ab32/SCAN_ALL_MRI_0_dataset.zip" --output "$OUTPUT_FILE"

# Optional: Extract the tarball if you want
if [ -f "$OUTPUT_FILE" ]; then
    echo "Download complete. Extracting..."
    tar -xvzf "$OUTPUT_FILE"
    echo "Extraction done."
else
    echo "Download failed or incomplete."
fi
