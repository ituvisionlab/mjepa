#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --job-name=tarextract
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gozde.unal@nyulangone.org
#SBATCH --output=/gpfs/data/sodicksonlab/gozde/slurm/download_fastmri_%j.log

# Optional: load modules if curl is not available by default
# module load curl

# Target directory
TARGET_DIR="/gpfs/data/sodicksonlab/gozde/LDM100K"

# File name for output
#OUTPUT_FILE="brain_fastMRI_DICOM.tar.gz"
#OUTPUT_FILE="fastmri_prostate_DICOMS_IDS_001_312.tar.gz"
#OUTPUT_FILE="fastMRI_breast_IDS_001_150_DCM.tar.gz"
OUTPUT_FILE="LDM_100k.tar"
#OUTPUT_FILE="knee_DICOMs_batch1.tar.xz" 
#OUTPUT_FILE="knee_DICOMs_batch2.tar.xz"

# Create target directory if not exists
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Extract the tarball if you want
if [ -f "$OUTPUT_FILE" ]; then
    echo "Extracting from tar file under "
    echo "Current directory: $(pwd)"
    tar -tvf "$OUTPUT_FILE"
    echo " Extraction done."
else
    echo "Download failed or incomplete."
fi
