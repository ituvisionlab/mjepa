# Data Generation Pipeline

This document describes the full preprocessing and data generation pipeline used to prepare MRI datasets for pretraining and downstream evaluation in the mjepa framework.

---

## Overview

The pipeline takes raw NIfTI (.nii.gz) files from multiple neuroimaging datasets and produces clean, analysis-ready CSV files containing file paths, labels, demographics, and bounding box coordinates. The pipeline is designed to be incremental — it can be re-run safely without reprocessing already handled subjects.

**Datasets covered:**
- SCAN (Systematic Characterization of Brain Alzheimer's and Related diseases through Neuroimaging)
- OASIS-3 (Open Access Series of Imaging Studies)
- ADNI (Alzheimer's Disease Neuroimaging Initiative)
- ABIDE (Autism Brain Imaging Data Exchange)
- BraTS 2024 (Brain Tumor Segmentation Challenge)
- PPMI (Parkinson's Progression Markers Initiative)
- IXI (Information eXtraction from Images)
- MOOD (Medical Out-of-Distribution)

---

## Common Preprocessing Steps (All Datasets)

### 1. Skull Stripping (BET)
All datasets undergo skull stripping using FSL BET (Brain Extraction Tool):
```
bet <input.nii.gz> <output_betmask.nii.gz> -m
```
- `-m` flag produces both the skull-stripped image and a binary brain mask
- Output files: `{base}_betmask.nii.gz` and `{base}_betmask_mask.nii.gz`
- BET is skipped if output files already exist (incremental processing)

### 2. Bounding Box Calculation
After BET, a tight bounding box is computed from the binary brain mask:
- All nonzero voxel coordinates are found using `np.argwhere(mask > 0)`
- Min/max coordinates are stored per axis: `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`
- Used downstream for brain-region cropping during training

### 3. Quality Filters Applied to All Datasets

| Filter | Criterion | Reason |
|--------|-----------|--------|
| File size | > 1 MB | Exclude corrupt or empty files |
| Dimensionality | Must be 3D | Exclude 4D or corrupted volumes |
| Field of View | > 50mm per axis | Exclude partial or scout scans |
| Voxel spacing | ≤ 6.5mm per axis | Exclude very low resolution scans |

### 4. Orientation Standardization
All volumes are reoriented to **RAS** (Right-Anterior-Superior) standard space using nibabel:
- Detects current orientation from the affine matrix
- Applies orientation transform to align with RAS convention
- Ensures consistent anatomical axis ordering across scanners and sites
- This also informs the correct axis for random horizontal flip augmentation

### 5. Output CSV Schema
All datasets produce CSVs with the following columns:

| Column | Description |
|--------|-------------|
| `label` | Diagnostic label (dataset-specific; -1 for pretraining-only subjects) |
| `subject_id` | Dataset-specific subject identifier |
| `contrast` | MRI contrast type (T1, T2, FLAIR, MPRAGE, etc.) |
| `date_acquired` | Scan date (YYYY-MM-DD) |
| `subject_sex` | M / F / N/A |
| `subject_age` | Approximate age (2025 - birth year) |
| `subject_weight` | Weight in kg (converted from lbs where applicable) |
| `nii_file_path` | Path to skull-stripped NIfTI file |
| `xmin/xmax/ymin/ymax/zmin/zmax` | Brain bounding box in voxel space |

---

## Dataset-Specific Preprocessing

### SCAN

**Source:** `SCAN_NIFTI/` directory with per-subject subdirectories

**Metadata:** `investigator_nacc69.csv`
- Columns used: `NACCID`, `NACCUDSD`, `SEX`, `BIRTHYR`, `WEIGHT`, `VISITYR`, `VISITMO`, `VISITDAY`
- Subject IDs zero-padded to 9 digits for consistent matching
- Visit date constructed from `VISITYR`, `VISITMO`, `VISITDAY` fields

**Label mapping (NACCUDSD):**
| Value | Meaning |
|-------|---------|
| 1 | Normal cognition |
| 2 | Impaired, not MCI |
| 3 | MCI |
| 4 | Dementia |

**Metadata matching:**
- Scan date extracted from filename via regex `(\d{4}-\d{2}-\d{2})`
- Each scan matched to the closest visit record by date (minimum `|VISITDATE - scan_date|`)
- Scans with missing or invalid `NACCUDSD` are excluded

**Contrast filtering:**
- Valid contrasts: `T1`, `T2`, `MPRAGE`, `FLAIR`, `_IR_`
- Contrast labels cleaned and standardized from filenames using keyword matching:

| Keyword in filename | Assigned contrast |
|--------------------|-------------------|
| `flair` | FLAIR |
| `mprage` | MPRAGE |
| `t2` | T2 |
| `\bt1\b` (word boundary) | T1 |
| `ir` or `fspgr` | IR |
| `gre` | GRE |
| `space` | SPACE |
| `star` | STAR |
| other | OTHER |

Note: Word boundary regex `\bt1\b` is used to avoid matching `t1` inside `t10`, `t12`, etc.

**Additional SCAN-specific filters:**
- `_ME_` sequences removed (Multi-Echo)
- `_3TE_` sequences removed (case-insensitive)

**Pretraining-only subjects:**
- Subjects with no metadata match (missing from `investigator_nacc69.csv`) are assigned `label=-1`
- These are included in pretraining CSV only, not in downstream evaluation

**SCAN pipeline order:**
```
Raw NIfTI files
      ↓
Basic CSV generation (label, metadata matching)
      ↓
BET + bbox + quality filtering → SCAN_NIFTI_all_with_betmask_and_bbox.csv
      ↓
Contrast label cleaning → SCAN_NIFTI_all_with_cleaned_contrast.csv
      ↓
Filename-based filter (_ME_, _3TE_) → SCAN_NIFTI_all_filtered.csv
      ↓
Intensity QC: compute mean/std per volume
      ↓
Intensity filter (mean > 2000 or std < 10 → excluded) → SCAN_NIFTI_filtered_by_stats.csv
      ↓
Final SCAN CSV
```

**Intensity filter thresholds:**
| Threshold | Value | Reason |
|-----------|-------|--------|
| MAX_ALLOWED_MEAN | 2000.0 | Quantitative map sequences have very high mean |
| MIN_ALLOWED_STD | 10.0 | Near-constant volumes are likely empty or artifact |

---

### OASIS-3

**Source:** `OASIS3all/` directory (BIDS-style structure)

**CSV Generation:**
- Walks all subject folders recursively for `.nii.gz` files
- Skips excluded subfolders: `fmap`, `func`, `swi`, `dwi`
- Skips files smaller than 2.5MB
- Applies FOV/spacing/dimensionality filters
- Reads per-scan JSON sidecar files for `SeriesDescription` metadata

**Excluded contrasts:**
`mIP`, `bold`, `SWI`, `unknown`, `DTI`, `ASL`, `dwi`, `field_mapping`, `bold-rest`, `epd2d`, `MDDW`, `rsfmri`, `Mag_`, `Pha_`

**Contrast mapping:**
| Keyword | Assigned contrast |
|---------|-------------------|
| T1 | T1 |
| T2 | T2 |
| MPRAGE | MPRAGE |
| FLAIR | FLAIR |
| T2_star | T2star |
| TOF | TOF |
| tse | tse |

**Label assignment (CDR-based):**
- Labels derived from `OASIS3_UDSb4_cdr.csv` using `CDRTOT` (Clinical Dementia Rating total score)
- Each scan matched to closest cognitive exam by `days_to_visit`
- Subject ID and scan day extracted from file path

**Post-processing filter:**
- TOF (Time of Flight) sequences removed — angiography sequences, not structural MRI
- Filter: rows where `contrast == "TOF"` are excluded → `oasis3_all_bet.csv`

---

### ADNI

**Source:** `ADNIall/ADNI/` directory with per-subject subdirectories; metadata in XML format per scan

**CSV Generation:**
- Walks all subject folders recursively for `.nii` files (not `.nii.gz`)
- Skips files smaller than 2.5MB
- Applies FOV/spacing/dimensionality filters (same as other datasets)
- Extracts metadata from per-scan XML files using the LONI IDA namespace

**XML metadata fields extracted:**
| Field | XML element | Description |
|-------|-------------|-------------|
| Label | `researchGroup` | Diagnostic group |
| Subject ID | `subjectIdentifier` | ADNI subject ID |
| Contrast | `protocol[@term="Weighting"]` | MRI weighting/contrast |
| Date | `dateAcquired` | Scan acquisition date |
| Sex | `subjectSex` | Subject sex |
| Age | `subjectAge` | Age at scan |
| Weight | `weightKg` | Weight in kg |

**Label mapping:**
| Label text | Value |
|------------|-------|
| CN (Cognitively Normal) | 0 |
| MCI | 1 |
| AD | 2 |
| EMCI | 3 |
| LMCI | 4 |
| Other | 5 |

**Downstream task preparation:**

Three task configurations are supported:

*Binary NC vs AD:*
- Keep labels `[0, 2]`, discard MCI
- Remap label `2 → 1` → (NC=0, AD=1)

*3-class NC vs MCI vs AD:*
- Keep labels `[0, 1, 2]`
- No remapping needed

*Binary NC vs MCI:*
- Keep labels `[0, 1]`
- No remapping needed

**Class imbalance handling (oversampling):**
- AD class is oversampled using `sklearn.utils.resample` with replacement
- Target count = number of NC samples
- Oversampled train sets saved separately (suffix `_oversampled.csv`)
- Validation sets are never oversampled

**Subject-level splitting:**
- Train/val split done at subject level to prevent leakage
- `train_test_split` on unique subject IDs, then rows filtered by subject membership

---

### PPMI

**Source:** `PPMI/` directory with per-subject subdirectories; metadata in XML format per scan

**CSV Generation (all contrasts):**
- Same XML-based metadata extraction as ADNI (same LONI IDA namespace)
- Subject ID, modality, study date, and image ID extracted from file path structure
- XML matched using pattern: `PPMI_{subject_id}_{modality}_S{study_id}_I{image_id}.xml`
- Applies FOV/spacing/dimensionality filters and file size filter (>2.5MB)

**CSV Generation (T1-anatomical only):**
- Restricts to folders containing `T1-anatomical` in path
- Skips `T2_in_T1-anatomical_space` folders explicitly
- Label extracted from XML using string search for `<researchGroup>` tag

**Label mapping:**
| Label text | Value |
|------------|-------|
| Control | 0 |
| PD (Parkinson's Disease) | 1 |
| Prodromal | 2 |
| SWEDD | 3 |
| Other | 4 |

---

### ABIDE

Same BET + bbox + quality filtering pipeline as SCAN. No dataset-specific label remapping — labels come directly from the ABIDE master CSV.

---

### BraTS 2020

**Source:** `BraTS2020_TrainingData/` and `BraTS2020_ValidationData/` directories

**CSV Generation:**
- Walks per-subject folders in both train and validation directories
- Extracts only **T1** modality (`t1.nii`) from each subject folder
- Subject ID extracted from folder name (last `_`-separated token)
- All subjects assigned `label=0` (pretraining only, no tumor grade labels used)

**Output columns:** `label`, `subject_id`, `nii_file_path`

---

### BraTS 2024

**Source:** `training_data1_v2/` and `training_data_additional/` directories

**CSV Generation:**
- Walks both training data folders recursively
- Excludes segmentation files (`-seg.nii.gz`)
- Computes bounding box from nonzero intensities (no BET needed — already skull-stripped)
- All subjects assigned `label=0` (pretraining only)

**Contrast mapping from filename:**
| Filename keyword | Assigned contrast |
|-----------------|-------------------|
| `t1c` | T1 |
| `t1n` | T1 |
| `t2w` | T2 |
| `t2f` | T2 |
| other | Unknown |

**Subject ID extraction:** Third token from filename split by `-`
- Example: `BraTS-GLI-02597-100-t1n.nii.gz` → subject ID `02597`

**Date acquired:** Last 3-digit number from filename used as placeholder session ID

**Scan UID extraction (for downstream use):** A unique scan identifier is extracted from the file path using dataset-aware regex patterns:

| Dataset style | Pattern | Example |
|--------------|---------|---------|
| ADNI/PPMI | `I\d+` | `I358089` |
| BraTS | `BraTS-[A-Za-z]+-\d{5}-\d{3}` | `BraTS-GLI-02795-100` |
| OASIS3 | Subject + session from BIDS filename | `OAS31092_d0203_T2w` |
| Fallback | File stem | filename without extension |

---

### IXI

**Source:** `IXI/` directory with recursive NIfTI discovery

**CSV Generation (basic):**
- Finds all `.nii.gz` files recursively using `glob`
- Subject ID extracted from filename: `IXI159-...` → `159`
- All subjects assigned `label=0`, `contrast=T1` (IXI is T1-only)
- No metadata available — sex, age, weight set to placeholder values

**CSV Generation (with BET + bbox):**
- Same as basic but additionally runs BET and computes bounding box
- If BET mask is missing or empty, falls back to full volume dimensions as bbox
- Output: `ixi_all_nii_with_bbox.csv`

---

### MOOD

**Source:** `MOOD/brain_train/` directory

**CSV Generation (basic):**
- Lists all files in the brain training folder (sorted)
- Subject ID extracted from filename stem
- All subjects assigned `label=0` (anomaly detection dataset, no class labels used)
- No BET or bbox computation — used for pretraining only

**CSV Generation (with bbox):**
- Finds all `.nii.gz` files recursively using `glob`
- Bounding box computed directly from nonzero voxel intensities (no BET needed)
- If no nonzero voxels found, falls back to full volume dimensions
- All contrasts assumed T1
- Output: `mood_all_nii_with_bbox.csv`

---

### ADNI (Additional — MD5 Deduplication)

When combining ADNI data from multiple local drives, duplicate files are detected and removed using MD5 checksums:
- Walks multiple source directories (`ADNI_NC`, `ADNI_NC1`, ..., `ADNI_NC4`)
- Computes MD5 hash for each `.nii` file
- Skips files whose hash has already been seen
- Outputs deduplicated `subject_id` and `nii_file_path` pairs to CSV

---

## General Filtering Steps (Cross-Dataset)

### Spatially Normalized Volume Removal
Volumes with `Spatially_Normalized` in their file path are excluded:
- These volumes are in MNI template space, not native subject space
- Mixing native and template space volumes introduces inconsistency
- Applied before other filters when combining datasets

### Spatial Dimension Filter
Only volumes with exactly **256 slices along the z-axis** (axis 2) are retained:
- Ensures consistent input dimensions for batching
- Avoids padding/cropping inconsistencies during training

---

## Quality Control

### Intensity Statistics
After preprocessing, per-volume intensity statistics are computed:
- `mean_intensity`: Mean voxel value across the brain volume (NaN/inf excluded)
- `std_intensity`: Standard deviation of voxel intensities

Saved to `volume_intensity_stats.csv` for review. Useful for detecting:
- Abnormally high mean (quantitative maps, scanner artifacts)
- Near-zero std (empty or constant volumes)
- Corrupt files with NaN/inf values

---

## Split Generation

After preprocessing, datasets are split for downstream evaluation:
- **Stratified K-fold** (5-fold) for ADNI and SCAN
- Splits are generated separately for each task (NC vs AD, NC vs MCI)
- Subject-level splitting — all scans from a subject stay in the same fold

### Overlap Checking
After splits are generated, subject-level overlap between pretrain and downstream sets is verified:
- Overlap checked at three levels: subject ID, basename, and subject+contrast+date
- Subjects appearing in pretrain are fully removed from downstream evaluation sets
- Ensures no data leakage between self-supervised pretraining and downstream evaluation

---

## Dependencies

- `nibabel` — NIfTI file loading and orientation handling
- `pandas`, `numpy` — data processing
- `FSL BET` — skull stripping (must be installed and on PATH)
- `scikit-learn` — stratified K-fold splitting
- `subprocess` — BET invocation

---

## Notes

- Subject age is approximated as `2025 - BIRTHYR` — not exact scan-time age
- Visit-to-scan matching uses closest date, not exact match — verify delta distribution for QC
- All preprocessing scripts are incremental and safe to re-run
- Logging is written per script to `.log` files for traceability

