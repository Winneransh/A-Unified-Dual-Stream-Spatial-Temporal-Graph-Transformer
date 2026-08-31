# A Unified Dual-Stream Spatial-Temporal Graph Transformer for Multi-Disease Neurological Disorder Detection

## Introduction
This is the official implementation of our paper "A Unified Dual-Stream Spatial-Temporal Graph Transformer for Multi-Disease Neurological Disorder Detection and Diagnosis from EEG Signals via Fractal Neurophysiological Biomarkers." The repository contains the preprocessing, training, and evaluation scripts used to reproduce the results reported in the paper.

## Datasets
Six publicly available EEG datasets were used across four neurological conditions (schizophrenia, Alzheimer's disease, frontotemporal dementia, and Parkinson's disease). Each dataset is obtainable from its original public repository.

| Dataset | Condition | Source |
|---|---|---|
| AHEPA | AD / FTD | [OpenNeuro (ds004504)](https://openneuro.org/datasets/ds004504) |
| Warsaw (RepOD) | SZ | [RepOD](http://dx.doi.org/10.18150/repod.0107441) |
| SU-SZ | SZ | [Zenodo](https://doi.org/10.5281/zenodo.14808296) |
| UC San Diego | PD | [OpenNeuro (ds002778)](https://openneuro.org/datasets/ds002778) |
| UNM | PD | [Narayanan Lab (Anjum et al.)](http://bit.ly/2QD3N7j) |
| UI | PD | [Narayanan Lab (Anjum et al.)](http://bit.ly/2QD3N7j) |

## Results

**Overall 10-fold cross-validation:** $98.16 \pm 0.21\%$ five-class accuracy (99.54% SZ, 98.34% AD/FTD, 99.56% PD).

**Leave-one-subject-out (LOSO):** 87.00% (AD/FTD), $87.00 \pm 2.62\%$ (SZ), 86.00% (PD).

**Leave-one-dataset-out (LODO):** 77–85% (mean 81.08%) across all cross-dataset configurations.

| Method | Approach | Dataset | Acc. (%) |
|---|---|---|---|
| Gosala et al. | GCN-LSTM | Warsaw | 99.25 |
| Shoeibi et al. | 1D Transformer | Warsaw | 97.62 |
| Decision Tree | Ensemble | SU-SZ | 93.81 |
| **Proposed** | Dual-Stream GCN-Trans. + Fractal | Warsaw | **99.54** |
| **Proposed** | Dual-Stream GCN-Trans. + Fractal | SU-SZ | **99.17** |
| Lalawat et al. (NeuroFormer) | Transformer | AHEPA | 95.29 |
| **Proposed** | Dual-Stream GCN-Trans. | AHEPA | **98.34** |
| Mukherjee \& Roy (DeePD-Net) | IM-CEEMDAN + DL | UC San Diego | 99.31 |
| Anuraj \& Menon | GSP + ViT | UNM | 98.61 |
| Anuraj \& Menon | GSP + ViT | UI | 99.11 |
| **Proposed** | Dual-Stream GCN-Trans. + Fractal | UC San Diego | **99.56** |
| **Proposed** | Dual-Stream GCN-Trans. + Fractal | UNM | **99.61** |
| **Proposed** | Dual-Stream GCN-Trans. + Fractal | UI | **99.48** |
