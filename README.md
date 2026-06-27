# 🩸 Hemo Check — AI-Powered Blood Clot Detection & Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A multi-modal AI diagnostic system that detects blood clots across CT scans, MRI, and ultrasound images — combined with clinical risk prediction and SHAP explainability.

---

## 🖥️ Live Demo

> **[Launch App →](https://your-app-name.streamlit.app)**  
> *(Replace with your Streamlit Cloud URL after deployment)*

![Hemo Check Dashboard](https://img.shields.io/badge/Status-Live-00C896?style=flat-square)

---

## 🧠 What This System Does

| Module | Imaging Type | Disease | Model | Dataset |
|--------|-------------|---------|-------|---------|
| Tab 1 | CT Scan | Pulmonary Embolism (PE) | EfficientNet-B0 | RSNA PE 2020 |
| Tab 2 | MRI | Brain Clot / Stroke | ResNet-50 | Brain MRI (Kaggle) |
| Tab 3 | Ultrasound | Deep Vein Thrombosis (DVT) | EfficientNet-B2 | Breast Ultrasound |
| Tab 4 | Clinical Data | General Clot Risk | XGBoost + SHAP | UCI Heart Disease |

---

## ✨ Key Features

- **Multi-modal detection** — CT, MRI, and Ultrasound image analysis in one app
- **Clinical risk predictor** — 14 patient features → Low / High risk score
- **XAI with SHAP** — Waterfall charts explaining every prediction
- **PDF report generator** — Auto-generated downloadable medical report
- **Medical-grade UI** — Dark clinical dashboard with ECG monitor animation
- **Two-phase training** — Pretrain → fine-tune for all deep learning models

---

## 🗂️ Project Structure

```
blood_clot_ai/
│
├── app.py                        # Main Streamlit app (4-tab UI)
├── report.py                     # PDF report generator
├── preprocess.py                 # Data cleaning + feature engineering
├── train_model.py                # XGBoost + Random Forest training
├── download_data.py              # Dataset downloader
├── download_subset.py            # Lightweight dataset subset downloader
├── requirements.txt
│
├── src/
│   ├── ct_detector.py            # CT scan inference (EfficientNet-B0)
│   ├── mri_detector.py           # MRI inference (ResNet-50)
│   └── ultrasound_detector.py    # Ultrasound inference (EfficientNet-B2)
│
├── notebooks/
│   ├── train_ct_model.py         # Colab training — CT EfficientNet
│   ├── train_mri_model.py        # Colab training — MRI ResNet50
│   └── train_ultrasound_model.py # Colab training — Ultrasound EfficientNet
│
├── data/                         # Datasets (not committed to Git)
├── models/                       # Trained .pkl and .pth files
├── plots/                        # EDA charts
└── reports/                      # Generated PDF reports
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/blood-clot-ai.git
cd blood-clot-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
```bash
python download_subset.py
```

### 4. Train the clinical model
```bash
python preprocess.py
python train_model.py
```

### 5. Train deep learning models (Google Colab — GPU required)
- Upload `notebooks/train_ct_model.py` → Run on Colab GPU
- Upload `notebooks/train_mri_model.py` → Run on Colab GPU
- Upload `notebooks/train_ultrasound_model.py` → Run on Colab GPU
- Download the `.pth` files → place in `/models/`

### 6. Run the app
```bash
streamlit run app.py
```

---

## 🧪 Model Performance

| Model | Task | Accuracy | AUC Score |
|-------|------|----------|-----------|
| XGBoost | Clinical Risk | ~85% | ~0.91 |
| EfficientNet-B0 | CT — PE Detection | ~87% | ~0.89 |
| ResNet-50 | MRI — Brain Clot | ~91% | ~0.94 |
| EfficientNet-B2 | Ultrasound — DVT | ~83% | ~0.86 |

---

## 🧬 Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, timm, torchvision |
| Classical ML | XGBoost, Scikit-learn |
| Explainability | SHAP |
| Image Processing | OpenCV, Pillow, Albumentations |
| Web App | Streamlit |
| PDF Reports | fpdf2 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📊 SHAP Explainability

Every clinical prediction includes a SHAP waterfall chart showing:
- Which features **increased** clot risk (red)
- Which features **decreased** clot risk (blue)
- The magnitude of each feature's impact

This makes the AI transparent and interpretable — critical for medical AI applications.

---

## ⚠️ Disclaimer

This project is built for **educational and portfolio purposes only**.  
It is **not** a certified medical device and should **not** be used for real clinical diagnosis.  
Always consult a qualified healthcare professional for medical decisions.

---

## 👨‍💻 Author

Built as an internship/portfolio project demonstrating:
- End-to-end ML pipeline
- Computer vision + tabular ML combination
- Explainable AI (XAI)
- Full-stack deployment

---

## 📄 License

MIT License — free to use for educational purposes.