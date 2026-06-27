# Hemo Check — AI-Powered Blood Clot Detection & Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat-square)
![fpdf2](https://img.shields.io/badge/fpdf2-2.7-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A multi-modal AI diagnostic system that detects blood clots from CT scans, MRI, and
> ultrasound images — combined with clinical risk prediction, Grad-CAM heatmaps,
> SHAP explainability, and downloadable PDF reports.

---

## Live Demo

> **[Launch App on Streamlit Cloud](https://your-app-name.streamlit.app)**
> *(Replace with your deployed URL)*

---

## What Hemo Check Does

Each scan tab (CT / MRI / Ultrasound) runs a **two-model combined pipeline**:

| Step | What happens |
|------|-------------|
| 1 | User uploads a scan image |
| 2 | User fills in patient clinical details (age, BP, cholesterol, symptoms, etc.) |
| 3 | Deep learning scan model runs (EfficientNet / ResNet) |
| 4 | Clinical ensemble runs (XGBoost + Random Forest soft voting) |
| 5 | Symptom severity scored with rule-based scorer |
| 6 | All three fused into one **Combined AI Assessment** with overall risk score |
| 7 | Grad-CAM heatmap generated showing model attention on the scan |
| 8 | SHAP waterfall chart explains clinical feature impact |
| 9 | Full combined PDF report downloaded — scan + clinical + recommendations |
| 10 | Prediction saved to SQLite history and audit trail |

---

## Diagnostic Modules

| Tab | Imaging Type | Disease | Model | Dataset |
|-----|-------------|---------|-------|---------|
| CT Scan | Chest CT | Pulmonary Embolism | EfficientNet-B0 | RSNA PE 2020 |
| MRI | Brain MRI | Brain Clot / Stroke | ResNet-50 | Brain MRI (Kaggle) |
| Ultrasound | Venous Ultrasound | DVT | EfficientNet-B2 | Breast Ultrasound |
| Clinical Risk | Patient data | General Clot Risk | XGBoost + RF | UCI Heart Disease |

---

## Key Features

- **Combined AI assessment** — scan + clinical + symptom score fused into one risk score
- **Grad-CAM heatmaps** — visual attention overlay on every scan image
- **SHAP explainability** — waterfall charts per clinical prediction
- **Ensemble AI** — XGBoost + Random Forest soft voting with disagreement flagging
- **Confidence thresholding** — below 70% flagged as inconclusive
- **Patient history** — SQLite-backed prediction history with risk trend charts
- **Full audit trail** — every inference logged with timestamp, model version, physician
- **Combined PDF reports** — scan result + clinical assessment + SHAP + Grad-CAM + recs
- **Unicode-safe PDF** — DejaVu font downloaded automatically for full character support
- **MediLab-style UI** — DM Sans + DM Serif Display, teal gradient hero, white cards

---

## Project Structure

```
blood_clot_ai/
|
|-- app.py                         # Main Streamlit app (8 tabs)
|-- report.py                      # Unicode-safe PDF report generator
|-- preprocess.py                  # Data cleaning + feature engineering
|-- train_model.py                 # XGBoost + Random Forest training
|-- download_data.py               # Dataset downloader (OpenML)
|-- download_subset.py             # Kaggle subset downloader
|-- requirements.txt
|-- README.md
|
|-- src/
|   |-- clinical_panel.py          # Reusable clinical form + predictor
|   |-- combined_analysis.py       # Scan + clinical + symptom fusion
|   |-- ct_detector.py             # CT scan inference (EfficientNet-B0)
|   |-- mri_detector.py            # MRI inference (ResNet-50)
|   |-- ultrasound_detector.py     # Ultrasound inference (EfficientNet-B2)
|   |-- ensemble.py                # XGBoost + RF soft voting ensemble
|   |-- gradcam.py                 # Grad-CAM heatmap generator
|   |-- history.py                 # SQLite prediction history
|   |-- audit.py                   # CSV audit trail logger
|   |-- analytics.py               # ROC, confusion matrix, feature importance
|
|-- notebooks/
|   |-- train_ct_model.py          # Colab training - CT EfficientNet-B0
|   |-- train_mri_model.py         # Colab training - MRI ResNet-50
|   |-- train_ultrasound_model.py  # Colab training - Ultrasound EfficientNet-B2
|
|-- data/                          # Datasets (not committed to Git)
|-- models/                        # Trained .pkl and .pth files
|-- fonts/                         # DejaVu fonts (auto-downloaded)
|-- plots/                         # EDA and analytics charts
|-- reports/                       # Generated PDF reports
```

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/blood-clot-ai.git
cd blood-clot-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the clinical dataset
```bash
python download_data.py
```

### 4. Preprocess and train the clinical model
```bash
python preprocess.py
python train_model.py
```

### 5. Train deep learning models (Google Colab — GPU required)
- Upload `notebooks/train_ct_model.py` to Colab, set Runtime to GPU, run
- Upload `notebooks/train_mri_model.py`, run
- Upload `notebooks/train_ultrasound_model.py`, run
- Download the saved `.pth` files and place them in `/models/`

### 6. Run the app
```bash
streamlit run app.py
```

The app auto-downloads DejaVu Unicode fonts on first run for PDF support.

---

## Model Performance

| Model | Task | Accuracy | AUC |
|-------|------|----------|-----|
| XGBoost | Clinical Risk | ~85% | ~0.91 |
| Random Forest | Clinical Risk | ~83% | ~0.88 |
| Ensemble (XGB+RF) | Clinical Risk | ~86% | ~0.92 |
| EfficientNet-B0 | CT - PE Detection | ~87% | ~0.89 |
| ResNet-50 | MRI - Brain Clot | ~91% | ~0.94 |
| EfficientNet-B2 | Ultrasound - DVT | ~83% | ~0.86 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, timm, torchvision |
| Classical ML | XGBoost, Scikit-learn |
| Explainability | SHAP, Grad-CAM |
| Image Processing | OpenCV, Pillow, Albumentations |
| Web App | Streamlit |
| PDF Reports | fpdf2 + DejaVu Unicode fonts |
| Database | SQLite (via Python sqlite3) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |

---

## Combined AI Assessment — Fusion Weights

| Source | Weight | Why |
|--------|--------|-----|
| Scan model | 50% | Direct imaging evidence is primary signal |
| Clinical ensemble | 35% | Clinical features are well-validated predictors |
| Symptom score | 15% | Supporting signal, not standalone diagnostic |

---

## Disclaimer

This project is built for **educational and portfolio purposes only**.
It is **not** a certified medical device and must **not** be used for real clinical
diagnosis. Always consult a qualified healthcare professional for medical decisions.

---

## Author

Built as a final year CSE project demonstrating:
- End-to-end ML pipeline design
- Computer vision + tabular ML integration
- Explainable AI (XAI) with SHAP and Grad-CAM
- Multi-model fusion and ensemble methods
- Full-stack deployment with Streamlit

---

## License

MIT License — free to use for educational purposes.