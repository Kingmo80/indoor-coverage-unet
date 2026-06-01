\# Indoor Coverage U-Net
### Machine Learning for Indoor Cellular Coverage Prediction Using Ray Tracing

> Predicts 5G/LTE indoor signal strength maps from building geometry in **under 1 second** — replacing hours of ray-tracing simulation.

---

## Overview

This project develops a deep learning system that learns to predict indoor wireless coverage maps directly from building geometry descriptions. Using **10,000 Sionna ray-tracing simulations** as training data, a **U-Net encoder–decoder** network predicts per-pixel received signal power at three height levels simultaneously.

A complementary classification model maps each spatial location to one of five coverage quality classes: **No Signal / Poor / Fair / Good / Excellent**.

| Model | MAE (dBm) | RMSE (dBm) | R² |
|---|---|---|---|
| FSPL Baseline | 30.338 | 31.230 | −8.505 |
| FSPL + Wall-Count | 73.431 | 88.097 | −74.640 |
| **U-Net (ours)** | **50.318** | **92.486** | **0.387** |

---

## Key Features

- 🏗️ **Procedural floorplan generator** — 4 room archetypes, 8 wall materials, 2 frequencies (700 MHz & 3.5 GHz)
- 📡 **Sionna ray-tracing pipeline** — 10,000 physically accurate simulations using NVIDIA Sionna
- 🧠 **U-Net regression model** — predicts received power (dBm) at 0.5 m, 1.5 m, and 2.5 m heights
- 🗺️ **U-Net classification model** — 5-class coverage quality map (balanced accuracy: 49.98%)
- ⚡ **Fast inference** — < 1 second on CPU, < 100 ms on GPU
- 🧑‍💻 **User-friendly notebook** — predict coverage for any custom room, no ML expertise needed

---

## Repository Structure

indoor-coverage-unet/
│
├── Research_Project_Machine_Learning_for_Indoor_Cellular_Coverage_Prediction_Using_Ray_Tracing_.ipynb
│   └── Main research notebook: data generation, model training, evaluation
│
├── USER_Predict_Coverage.ipynb
│   └── User-facing prediction notebook: define a room and get coverage maps in seconds
│
├── Indoor_Coverage_Report (2) (1).pdf
│   └── Full research report with methodology, results, and analysis
│
├── Indoor_Coverage_Predictor_Handbook (1) (2).pdf
│   └── Step-by-step user guide for the prediction notebook
│
├── LICENSE                  ← MIT License
└── README.md

---

## Getting Started

### Prerequisites

- Google Account with Colab access (Colab Pro recommended for GPU)
- Trained model files saved to Google Drive (download links below)

### Running the Prediction Notebook

1. Open USER_Predict_Coverage.ipynb in Google Colab
2. Mount your Google Drive and update the model path in Cell 1
3. Edit Cell 2 to define your room (walls, windows, furniture, transmitter)
4. Run Cells 3, 4, and 5 to get your coverage maps

### Model Files

The trained model files are too large for GitHub. Download them from Google Drive:

📁 Trained Models (part4_outputs):
https://drive.google.com/drive/folders/1VAVv6WPRo3ZQxtYv9zV4CAWrVt5naJFJ?usp=drive_link

📁 Processed Dataset (dataset_processed):
https://drive.google.com/drive/folders/1sFbJGqcXsm2r2jvi1-UOieUPbn9oIQEv?usp=drive_link

Place the model files at:
/content/drive/MyDrive/part4_outputs/coverage_unet.keras
/content/drive/MyDrive/part4_outputs/coverage_clf_unet.keras
/content/drive/MyDrive/dataset/dataset_processed/representation_metadata.json

---

## System Pipeline

Procedural      Sionna Ray      6-Channel        U-Net
Floorplans  →   Tracing     →   Tensor       →   Models  →  Coverage Map
(4 types)       (10,000)        Encoding          (< 1s)     (dBm + class)

### Input Tensor (6 channels, 0.5 m resolution)

| Channel | Description |
|---|---|
| 0 | Wall presence (binary) |
| 1 | Wall material (normalised index) |
| 2 | Window presence (binary) |
| 3 | Furniture presence (binary) |
| 4 | TX distance (normalised) |
| 5 | Frequency (0 = 700 MHz, 1 = 3.5 GHz) |

---

## Coverage Classes

| Class | dBm Range | Meaning |
|---|---|---|
| No Signal | < −154 dBm | No detectable signal |
| Poor | −154 to −60 dBm | Very weak; calls may drop |
| Fair | −60 to −50 dBm | Usable for voice, marginal for data |
| Good | −50 to −43 dBm | Reliable voice and moderate data |
| Excellent | > −43 dBm | Full signal; high data rates |

---

## Wall Materials Supported

| Material | Attenuation |
|---|---|
| concrete | ~20 dB |
| thick_concrete | ~30 dB |
| reinforced | ~35 dB |
| brick | ~15 dB |
| drywall | ~6 dB |
| glass | ~3 dB |
| wood | ~5 dB |
| metal | ~40 dB |

---

## Authors

**Zack Moyambo**
Texas A&M University – Central Texas

**Professor Christopher Thron** (Supervisor)
Texas A&M University – Central Texas

---

## Citation

If you use this work, please cite:

@article{moyambo2025indoor,
  title  = {Machine Learning for Indoor Cellular Coverage Prediction Using Ray-Tracing Simulation},
  author = {Moyambo, Zack and Thron, Christopher},
  year   = {2025}
}

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## Acknowledgements

- NVIDIA Sionna: https://github.com/NVlabs/sionna — ray-tracing simulation engine
- U-Net: https://arxiv.org/abs/1505.04597 — Ronneberger et al., MICCAI 2015
- ITU-R P.2040: https://www.itu.int/rec/R-REC-P.2040/en — building material electromagnetic properties
