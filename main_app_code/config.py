"""
PharmaCare — Central Configuration
All settings are defined here. Import from this file everywhere.
"""
import os

# ── Base directory (this file lives in main_app_code/) ─────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── File Paths ──────────────────────────────────────────────────────────────
DATA_XLSX   = os.path.join(BASE_DIR, "..", "data", "updated_ner_results.xlsx")
MODEL_PATH  = os.path.join(BASE_DIR, "model-best")

# ── Fraud Detection ─────────────────────────────────────────────────────────
FRAUD_THRESHOLD = 0.5          # Jaccard score ≥ this → Authentic

# ── OCR & Upload ────────────────────────────────────────────────────────────
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
MAX_FILE_SIZE_MB  = 15
MIN_TEXT_LENGTH   = 5          # minimum chars to proceed past OCR

# ── App Metadata ────────────────────────────────────────────────────────────
APP_NAME     = "PharmaCare"
APP_ICON     = "💊"
APP_SUBTITLE = "AI-Powered Medicine Fraud Detection"
APP_VERSION  = "2.1.0"
