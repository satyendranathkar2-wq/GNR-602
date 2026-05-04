# GNR-602 — Satellite Image Segmentation (ASIP Project)

A pixel-level semantic segmentation system for satellite imagery, built from scratch using pure NumPy and PyTorch (no `nn.Module`). The system trains a **single MLP baseline** and an **ensemble of MLPs** with different architectures, and ships a **Streamlit web app** for interactive inference.

---

## 📌 Project Overview

This project segments high-resolution satellite images into 6 land-cover classes:

| Label | Class | Color |
|-------|-------|-------|
| 0 | Building | `rgb(60, 16, 152)` |
| 1 | Land (unpaved) | `rgb(132, 41, 246)` |
| 2 | Road | `rgb(110, 193, 228)` |
| 3 | Vegetation | `rgb(254, 221, 58)` |
| 4 | Water | `rgb(226, 169, 41)` |
| 5 | Unlabeled | `rgb(155, 155, 155)` |

Each pixel in a satellite image is classified independently using its RGB value as input features — no convolutions, no pretrained encoders.

---

## 🗂️ Repository Structure

```
GNR-602/
├── colour_mapping/           # Saved colour ↔ label dictionaries (color_mappings.pkl)
├── model_weights/            # Trained model weights
│   ├── single_model.pkl      # Single MLP weights
│   └── ensemble_weights.pkl  # List of ensemble MLP weights
├── ppt_asip/                 # Project presentation slides
├── project_ipynb file and app/
│   ├── Asip_project.ipynb    # Full training pipeline notebook
│   └── app.py                # Streamlit inference app
├── test image accuracy/      # Sample test images with accuracy results
└── training curves and validation accuracy/  # Training loss & validation accuracy plots
```

---

## ⚙️ How It Works

### 1. Dataset
- **Source:** Satellite image dataset with paired RGB images and segmentation masks
- Images are resized to **256×256** pixels
- Each pixel becomes a training sample: `(R, G, B)` → `class label`
- Split: **70% train / 20% validation / 10% held-out test** (by full image pairs, not by pixel)

### 2. Model Architecture
Models are MLPs built entirely from scratch (no `nn.Module`):
- **Input:** 3 features (normalised RGB)
- **Hidden layers:** ReLU activations
- **Output:** Softmax over 6 classes

**Single model:** One MLP trained as the baseline.

**Ensemble:** Multiple MLPs with varying hidden layer sizes and node counts, combined via **majority vote** at inference time.

### 3. Training Details
| Hyperparameter | Value |
|---|---|
| Image size | 256 × 256 |
| Epochs | 15 |
| Learning rate | 0.0015 |
| LR decay | 0.90 per epoch |
| Mini-batch size | 16,384 pixels |
| Optimiser | Adam (from scratch) |
| Loss | Weighted cross-entropy (class weight power = 0.5) |
| Gradient clipping | 5.0 |
| Weight decay | 1e-4 |

### 4. Inference App
The Streamlit app (`app.py`) allows:
- Upload any satellite image
- View segmentation from **Single Model** and **Ensemble** side-by-side
- Optionally upload a **Ground Truth mask** to compute pixel accuracy

---

## 🚀 Running the App

### Prerequisites
```bash
pip install streamlit numpy torch pillow
```

### Required files (place in the same directory as `app.py`)
```
app.py
ensemble_weights.pkl
single_model.pkl
color_mappings.pkl
```

### Launch
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📊 Results

Training curves, validation accuracy plots, and test image comparisons are available in the respective folders in this repository.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **PyTorch** — tensor ops and autograd (no `nn.Module` used)
- **NumPy** — data processing
- **Pillow (PIL)** — image loading and resizing
- **Streamlit** — interactive web app
- **Matplotlib** — training visualisation

---

## 📁 Dataset

The model was trained on the [Satellite Dataset]((https://www.kaggle.com/datasets/hammadjavaid/multi-source-satellite-imagery-for-segmentation)) from Kaggle, consisting of paired satellite images and RGB-encoded segmentation masks.

---

## 👤 Author
**Kajal Kumari** |
**Neeraj Datar** |
**satyendranath kar**| 
IIT Bombay — GNR-602 Course Project
