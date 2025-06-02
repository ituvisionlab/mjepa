#!/bin/bash
#SBATCH --partition=cpu_medium
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --job-name=download
#SBATCH --mail-type=END
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/slurmDownload_%j.log



# Optional: load modules if curl is not available by default
# module load curl

# Target directory
TARGET_DIR="/gpfs/data/sodicksonlab/gozde/FastMRI"

# File name for output
OUTPUT_FILE="brain_fastMRI_DICOM.tar.gz"
#OUTPUT_FILE="fastmri_prostate_DICOMS_IDS_001_312.tar.gz"
#OUTPUT_FILE="fastMRI_breast_IDS_001_150_DCM.tar.gz"
#OUTPUT_FILE="fastMRI_breast_IDS_150_300_DCM.tar.gz"
#OUTPUT_FILE="knee_DICOMs_batch1.tar.xz" 
#OUTPUT_FILE="knee_DICOMs_batch2.tar.xz"

# Create target directory if not exists
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Actual download command with resume support
#curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastmri_prostate_DICOMS_IDS_001_312.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=YO7PB9Zf%2BTz6NpmlTejwKVUCxWw%3D&Expires=1745437976" --output "$OUTPUT_FILE"
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_fastMRI_DICOM.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=%2FtHezig3ju%2BKNvPTf%2By1PzJRxcE%3D&Expires=1745437976" --output "$OUTPUT_FILE"

# Optional: Extract the tarball if you want
if [ -f "$OUTPUT_FILE" ]; then
    echo "Download complete. Extracting..."
    tar -xvzf "$OUTPUT_FILE"
    echo "Extraction done."
else
    echo "Download failed or incomplete."
fi
