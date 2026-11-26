# ThermoAE-Training

Sparse Autoencoder training pipeline for unsupervised anomaly and fault detection in building thermal/heating systems.

This code was developed as part of an international research collaboration between the **Federal University of Lavras (UFLA, Brazil)** and the **University of the Basque Country (UPV/EHU, Bilbao, Spain)**, combining sparse autoencoders with thermoeconomic fault diagnosis — resulting in a **peer-reviewed article published in an A1 journal**:

- **Published Paper:** [ScienceDirect — Journal of Building Engineering](https://www.sciencedirect.com/science/article/abs/pii/S2352710226003190?via%3Dihub)
- **Companion Inference Repository:** [`Thestarki-ThermoAE_Inference`](https://github.com/Thestarki/Thestarki-ThermoAE_Inference)

## Overview

The model learns normal thermal-plant operating behavior from fault-free reference telemetry (`Dados/Reference_data_1min.xlsx`) and detects unlabeled anomalies across tank, boiler, and heat-exchanger fault scenarios (`Op_data_MF1_tank.xlsx`, `OP_data_MF2_tank.xlsx`, `OP_data_MF3.xlsx`, `Op_data_MF_boi.xlsx`, `Op_data_MF_hx.xlsx`) via reconstruction error.

## Repository Structure

```text
src/
├── RunnerTraining.py          # Training entrypoint
├── configs/config.py          # Training, model, and path settings
├── db/db_model.py             # SQLite experiment/artifact persistence
└── modelo/
    ├── ler_dados.py           # Dataset ingestion
    ├── tratar_dados.py        # Preprocessing and normalization
    ├── criar_dataloaders.py   # PyTorch DataLoader creation
    └── modelo.py              # Sparse Autoencoder architecture & training loop
```

## Tech Stack

Python · PyTorch · Pandas · scikit-learn · SQLite · Poetry

> **Note:** For newer, production-structured microservice refactors (FastAPI, Docker, Pydantic, pytest, CI) on related thermal-plant fault detection datasets, see [`ae-gbm-fault-classifier`](https://github.com/Thestarki/ae-gbm-fault-classifier) and [`autoencoder-tank-anomaly-detector`](https://github.com/Thestarki/autoencoder-tank-anomaly-detector).
