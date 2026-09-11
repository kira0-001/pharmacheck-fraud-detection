<div align="center">

# 💊 PharmaCare — AI-Powered Medicine Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit)
![Tesseract](https://img.shields.io/badge/OCR-Tesseract-green?style=for-the-badge)
![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=for-the-badge&logo=spacy)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Upload a medicine image → AI reads the label → Instantly detects if it is authentic or fraudulent**

[🚀 Live Demo](#) · [📖 How it Works](#how-it-works) · [⚙️ Setup](#setup)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Tesseract OCR** | Extracts text from medicine labels using Sparse Text (PSM 11) mode |
| 🧠 **Custom spaCy NER** | Tags Drug Name, Composition, Size, Dosage using a custom-trained model |
| 🛡️ **Fraud Detection** | Overlap Coefficient similarity scoring against a 477-record drug database |
| 🌍 **Multi-Language** | English, Spanish, French, German label support |
| 📄 **PDF Report** | Downloads a branded, color-coded fraud analysis report with the original image |
| 🗄️ **Admin Dashboard** | Built-in Developer Panel to view, search, and add medicines to the database |
| ✏️ **Auto-Corrector** | Spell-checker fixes OCR typos (`T@blet` → `Tablet`) before analysis |

---

## 🖼️ Screenshots

> Upload → Extract Text → NER Analysis → Fraud Verdict

---

## How it Works

```
📷 Upload Image
     ↓
🔍 Tesseract OCR (PSM 11 Sparse Text)
     ↓
✏️  Spell Correction (pyspellchecker)
     ↓
🧬 spaCy NER → Drug Name, Composition, Size, Dosage
     ↓
📊 Overlap Coefficient vs. 477 Drug Records
     ↓
🛡️  Verdict: AUTHENTIC (≥50%) or POTENTIALLY FRAUDULENT (<50%)
     ↓
📄 Download PDF Report
```

---

## Setup

### Prerequisites
- Python 3.11+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-pharmacheck.git
cd fraud-detection-pharmacheck/fraud_platform_imen
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download language packs (optional)
```bash
python setup_languages.py
```

### 5. Run the app
```bash
streamlit run main_app_code/pages/main.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
fraud_platform_imen/
├── main_app_code/
│   ├── pages/
│   │   └── main.py          # 🖥️  Streamlit UI (all 4 steps)
│   ├── model-best/          # 🧠  Custom spaCy NER model
│   ├── config.py            # ⚙️  Central configuration
│   └── ner_app.py           # 🔬  OCR, NER, Fraud Logic, PDF
├── data/
│   └── updated_ner_results.xlsx  # 🗄️  Drug database (477 records)
├── .streamlit/
│   └── config.toml          # 🎨  Theme & server config
├── requirements.txt
└── README.md
```

---

## 🧠 AI Models Used

| Component | Technology |
|---|---|
| OCR Engine | Tesseract 5.x (PSM 11 – Sparse Text) |
| NER Model | Custom spaCy model trained on pharmaceutical labels |
| Similarity | Overlap Coefficient (robust to dense label noise) |
| Spell Check | pyspellchecker |

---

## 🚀 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Set **Main file path** to: `fraud_platform_imen/main_app_code/pages/main.py`
5. Click **Deploy** — you get a live HTTPS link in minutes!

> ⚠️ Note: You must add Tesseract installation to your `packages.txt` for cloud deployment (already included in this repo).

---

## 👩‍💻 Developer Panel

Switch to **Database Admin** mode in the sidebar to:
- View all 477 drug records in a searchable table
- Add new medicines via a form (no Excel needed)
- Changes take effect on the next scan immediately

---

## 📄 License

MIT © 2026 — Built as a BCP1 final year project.
