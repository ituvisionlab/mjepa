# Masked and Predictive Self-Supervised Foundation Models for 3D Brain MRI

Esra Ergün, Hersh Chandarana, Dan Sodickson, Gözde Ünal

## Overview

This repository contains the official implementation of our paper on self-supervised foundation model pretraining for 3D brain MRI-based disease detection. We investigate and extend two major self-supervised pretraining paradigms:

- **MAE** (Masked Autoencoders) — reconstruction-based learning, extended with a novel **spectral-domain reconstruction loss** (SL-MAE) to enhance sensitivity to fine-grained anatomical structure
- **JEPA** (Joint Embedding Predictive Architecture) — predictive representation learning, extended with **variance–covariance regularization** (VCR-JEPA) to encourage distributed and decorrelated latent representations

Models are trained on heterogeneous single-contrast MRI volumes (T1- or T2-weighted) in a contrast-agnostic setting, with randomized native in-plane orientation sampling to encourage learning across diverse radiological perspectives.

---

## Key Contributions

- Systematic comparison of MAE and JEPA pretraining paradigms for 3D MRI disease detection
- Novel spectral-domain reconstruction loss for MAE (SL-MAE)
- Variance–covariance regularization for JEPA (VCR-JEPA)
- Evaluation across five downstream disease detection tasks
- Subject-level dataset splitting with public release of all dataset partitions to improve reproducibility and prevent data leakage

---

## Data Preparation

The full preprocessing and data generation pipeline for all supported 
datasets (ADNI, NACC/SCAN, OASIS-3, ABIDE, BraTS-24, PPMI, IXI, MOOD) is 
described in detail in [`src/datasets/data_generation.md`](src/datasets/data_generation.md).

This includes skull stripping, bounding box computation, quality 
filtering, contrast standardization, label mapping, and train/val 
split generation for each dataset.

## Dataset Splits

To support reproducibility and prevent data leakage, we publicly release all subject-level splits used in our experiments. All splits are based on **subject-level partitioning**, ensuring no subject appears across train, validation, and test sets.

### Pretraining Subject and Scan IDs
Subject and scan IDs used for pretraining are available under `src/datasets/pretrain_subject_ids/`, with one CSV per dataset:

- `adni_pretrain_ids.csv`
- `oasis3_pretrain_ids.csv`
- `ppmi_pretrain_ids.csv`
- `ixi_pretrain_ids.csv`
- `mood_pretrain_ids.csv`
- `brats24_pretrain_ids.csv`
- `scan_pretrain_ids.csv`

### Downstream Evaluation Splits
Train, validation, and test subject and scan IDs for all downstream tasks are available under `src/datasets/downstream_subject_ids/`. Splits are provided for:

- **ADNI** — NC vs AD and NC vs MCI (5-fold cross-validation)
- **NACC/SCAN** — NC vs AD and NC vs MCI
- **UCSF** — Tumor grade classification (Grade 2 vs 3-4, Grade 2-3 vs 4, multiclass)
- **ABIDE** — Normal vs Autism

Each CSV contains `subject_id` and `scan_uid` columns for each train/val/test split.

---

## Code Structure

```
.
├── app/
│   └── vjepa/                  # Pretraining loops (JEPA, VCR-JEPA)
├── evals/
│   ├── classification3d/
│   │   ├── train.py             # Fine-tuning / linear probing
│   │   └── eval.py              # Inference and evaluation
│   ├── mri_inference/
│   └── mri_tsne_frozen/
├── src/
│   ├── datasets/                # Dataset loaders and data managers
│   ├── models/                  # Model definitions (ViT, ResNet3D, etc.)
│   ├── masks/                   # Masking utilities
│   └── utils/                   # Shared utilities
└── configs/
    ├── pretrain/                # Pretraining configs
    └── evals/                   # Evaluation configs
```

---

## Pretraining and Finetuning Datasets

### Pretraining

| Dataset | Domain | Contrast | Subjects | Volumes |
|---------|--------|----------|----------|---------|
| ADNI | Brain | T1 | 1224 | 31347 |
| OASIS3 | Brain | T1, T2, FLAIR, T2* | 1376 | 7834 |
| PPMI | Brain | T1 | 356 | 1675 |
| IXI | Brain | T1 | 581 | 581 |
| MOOD | Brain | T1 | 800 | 800 |
| BraTS 2024 | Brain | T1, T2 | 731 | 6484 |
| NACC/SCAN | Brain | T1, T2, FLAIR | 3701 | 10060 |
| **Total** | | | **8769** | **58781** |

### Downstream Evaluation

| Dataset | Domain | Task | Contrast | Subjects | Volumes |
|---------|--------|------|----------|----------|---------|
| ADNI | Brain | NC/MCI and NC/AD | T1 | 496 | 5392 |
| NACC/SCAN | Brain | NC/MCI and NC/AD | T1, T2, FLAIR | 1167 | 3147 |
| UCSF | Brain | Tumor Grade (Low vs High) | T1, T2, FLAIR, ADC | 184 | 2024 |
| ABIDE | Brain | Normal vs Autism | T1 | 1109 | 1206 |
| **Total** | | | | **2956** | **11769** |

All dataset splits used in our experiments are publicly released under `src/datasets/` to support reproducibility and prevent data leakage.

---

## Setup

```bash
conda create -n mjepa python=3.9 pip
conda activate mjepa
python setup.py install
```

---

## Pretraining

### Local (single/multi-GPU)

```bash
python -m app.main \
  --fname configs/pretrain/vitb16_mri.yaml \
  --devices cuda:0 cuda:1
```

### Distributed (SLURM)

```bash
python -m app.main_distributed \
  --fname configs/pretrain/vitb16_mri3.1.yaml \
  --folder $path_to_logs \
  --partition $slurm_partition
```

---

## Evaluation

### Training a linear probe / fine-tuning

```bash
python -m evals.classification3d.train \
  --fname configs/evals/vitb16_mri_eval.yaml
```

---

## Acknowledgements

This codebase builds on [V-JEPA](https://github.com/facebookresearch/jepa) by Meta AI Research. We thank the authors for releasing their code.

---

## License

See the [LICENSE](./LICENSE) file for details.
