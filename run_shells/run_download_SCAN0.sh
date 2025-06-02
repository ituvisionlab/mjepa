#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --job-name=download
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurmDownload_%j.log

# Optional: load modules if curl is not available by default
# module load curl

# Target directory
TARGET_DIR="/gpfs/data/sodicksonlab/gozde/SCAN"

# File name for output
OUTPUT_FILE="SCAN_MRI_metadata.zip"
# OUTPUT_FILE="SCAN_MRI_NACC69.zip"

#UDS NP genetics including APOE: 
# "https://naccquickaccess.s3.amazonaws.com/investigator_nacc69.csv?AWSAccessKeyId=AKIAJQO3SKE7XG2R2ACQ&Signature=mVRQGR30lItGe3%2Fg3TCq60b9GoU%3D&Expires=1745954713" 
# SCAN MRI: 
#"https://naccquickaccess.s3.amazonaws.com/investigator_scan_mri_nacc69.zip?AWSAccessKeyId=AKIAJQO3SKE7XG2R2ACQ&Signature=YIFWRFFN5cdDcmXyXPP7Xhdukc0%3D&Expires=1745954703"

# Create target directory if not exists
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Actual download command with resume support
curl -C - "https://ida.loni.usc.edu/download/image-metadata?key=31e6328c-54ae-49d5-84c5-e29d0c119663&zip=SCAN_MRI_0_metadata.zip" --output "$OUTPUT_FILE"

# Optional: Extract the tarball if you want
if [ -f "$OUTPUT_FILE" ]; then
    echo "Download complete. Extracting..."
    tar -xvzf "$OUTPUT_FILE"
    echo "Extraction done."
else
    echo "Download failed or incomplete."
fi
