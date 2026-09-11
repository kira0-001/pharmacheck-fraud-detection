"""
PharmaCare · ner_app.py  —  AI Backend  v2.1  (Lightweight Edition)
OCR: Tesseract (CPU-only, no GPU, no PyTorch, ~150MB)
NER: Custom spaCy model
"""
import cv2
import numpy as np
import os
import tempfile
import nltk
import pandas as pd
from PIL import Image as PILImage
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import hashlib
from spacy import displacy
from fpdf import FPDF
import streamlit as st

try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    # Add domain specific terms
    spell.word_frequency.load_words(['paracetamol', 'mg', 'tablet', 'syrup', 'capsule', 'injection'])
    HAS_SPELLCHECK = True
except ImportError:
    HAS_SPELLCHECK = False

# ── Bootstrap NLTK on first run ─────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Config ──────────────────────────────────────────────────────────────────
from config import DATA_XLSX, MODEL_PATH, FRAUD_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
#  Cached Resources (loaded once per server session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _check_tesseract():
    """Verify Tesseract is installed; raise a helpful message if not."""
    import pytesseract
    
    # Check default Windows installation path and custom path
    win_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    custom_path = r'D:\bcp1\Tesseract-OCR\tesseract.exe'
    
    if os.name == 'nt':
        if os.path.exists(custom_path):
            pytesseract.pytesseract.tesseract_cmd = custom_path
        elif os.path.exists(win_path):
            pytesseract.pytesseract.tesseract_cmd = win_path

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        raise RuntimeError(
            "Tesseract OCR binary not found!\n\n"
            "• Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "• Linux:   sudo apt install tesseract-ocr\n"
            "• Mac:     brew install tesseract"
        )
    return pytesseract


@st.cache_resource(show_spinner=False)
def get_spacy_nlp():
    """Load the custom-trained spaCy NER model once and cache it."""
    import spacy
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"spaCy model not found at: {MODEL_PATH}\n"
            "Ensure the 'model-best' folder is inside main_app_code/."
        )
    return spacy.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_and_preprocess_database():
    """Load the drug database and pre-tokenise each row for Jaccard lookups."""
    if not os.path.exists(DATA_XLSX):
        raise FileNotFoundError(
            f"Drug database not found at: {DATA_XLSX}\n"
            "Ensure 'updated_ner_results.xlsx' is inside the data/ folder."
        )
    df = pd.read_excel(DATA_XLSX)

    for col in ['DOSAGE', 'DRUGNAME', 'SIZE', 'COMPOSITION', 'TYPE']:
        if col in df.columns:
            df[col] = df[col].fillna(' ')

    stop_words = set(stopwords.words("english"))
    punct      = {',', '.', ':', 'nan'}

    def _tokenise_row(row):
        text  = ' '.join(str(row.get(c, '')) for c in ['DRUGNAME', 'TYPE', 'COMPOSITION', 'SIZE', 'DOSAGE'])
        words = word_tokenize(text)
        return {t.lower() for t in words if t.lower() not in stop_words and t not in punct}

    df['token_set'] = df.apply(_tokenise_row, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  OCR  (Tesseract — lightweight, no GPU required)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality before OCR.
    Steps: grayscale → upscale small images → CLAHE contrast → mild sharpen.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale if image is small (Tesseract is more accurate on larger images)
    h, w = gray.shape
    if w < 1200:
        scale = 1200 / w
        gray  = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # CLAHE: boosts local contrast without over-brightening
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Mild sharpening kernel
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def ocr_extraction(image: np.ndarray, lang_code: str = 'eng'):
    """
    Run Tesseract OCR on a BGR numpy image.
    Returns: (full_text: str, bbox_result: list)
    """
    import pytesseract
    _check_tesseract()

    processed = _preprocess_for_ocr(image)
    pil_img   = PILImage.fromarray(processed)

    # PSM 11: Sparse text. Finds as much text as possible in no particular order.
    # OEM 3: Default LSTM.
    config = '--psm 11 --oem 3'
    text   = pytesseract.image_to_string(pil_img, lang=lang_code, config=config)

    # Clean text with spellchecker
    if HAS_SPELLCHECK:
        cleaned_words = []
        for word in text.split():
            # Only correct purely alphabetic words to avoid ruining numbers/codes
            if word.isalpha() and len(word) > 3:
                correction = spell.correction(word)
                cleaned_words.append(correction if correction else word)
            else:
                cleaned_words.append(word)
        text = ' '.join(cleaned_words)
        
    # Bounding boxes for visualization
    data   = pytesseract.image_to_data(pil_img, lang=lang_code, config=config,
                                        output_type=pytesseract.Output.DICT)
    result = []
    scale_x = image.shape[1] / processed.shape[1]   # map back to original coords
    scale_y = image.shape[0] / processed.shape[0]

    for i, word in enumerate(data['text']):
        word = word.strip()
        conf = int(data['conf'][i])
        if word and conf > 30:                       # only high-confidence words
            x = int(data['left'][i]   * scale_x)
            y = int(data['top'][i]    * scale_y)
            w = int(data['width'][i]  * scale_x)
            h = int(data['height'][i] * scale_y)
            result.append(
                ([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], word, conf / 100.0)
            )

    return text.strip(), result


def draw_contours(image: np.ndarray, result: list) -> np.ndarray:
    """Draw green bounding boxes around detected text regions."""
    for detection in result:
        top_left     = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        image = cv2.rectangle(image, top_left, bottom_right, (0, 180, 70), 2)
    return image


# ─────────────────────────────────────────────────────────────────────────────
#  Named Entity Recognition
# ─────────────────────────────────────────────────────────────────────────────

def perform_named_entity_recognition(text: str):
    """Run the custom spaCy NER pipeline on text. Returns a spaCy Doc."""
    nlp = get_spacy_nlp()
    return nlp(text)


def _stable_color(label: str) -> str:
    """Generate a deterministic pastel color based on the entity label string."""
    hash_obj = hashlib.md5(label.encode('utf-8'))
    hex_digest = hash_obj.hexdigest()
    
    # Generate RGB components, constraining to a pastel range (100-240)
    r = 100 + (int(hex_digest[0:2], 16) % 140)
    g = 100 + (int(hex_digest[2:4], 16) % 140)
    b = 100 + (int(hex_digest[4:6], 16) % 140)
    
    return f'#{r:02x}{g:02x}{b:02x}'


def display_doc(doc) -> str:
    """Render displacy HTML for the spaCy Doc."""
    colors  = {ent.label_: _stable_color(ent.label_) for ent in doc.ents}
    options = {"ents": list(colors.keys()), "colors": colors}
    return displacy.render(doc, style='ent', options=options, page=True, minify=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Summary / PDF
# ─────────────────────────────────────────────────────────────────────────────

def details_dict(doc) -> dict:
    """Convert a spaCy Doc into {label: [entity_strings]} dict."""
    result: dict = {}
    for ent in doc.ents:
        val = str(ent).strip()
        if not val:
            continue
        if ent.label_ not in result:
            result[ent.label_] = [val]
        elif val not in result[ent.label_]:
            result[ent.label_].append(val)
    return result


def create_summary_pdf(detail: dict, image_bytes: bytes = None, fraud_result: tuple = None) -> str:
    """Generate a branded, comprehensive PDF report. Returns PDF path."""
    from datetime import datetime
    
    def _sanitize(text: str) -> str:
        """Replace common unicode characters that break FPDF's latin-1 encoding."""
        if not text: return ""
        replacements = {
            '\u2014': '-', '\u2013': '-', '\u201c': '"', '\u201d': '"',
            '\u2018': "'", '\u2019': "'", '\u2026': '...'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # Encode to latin-1 and replace unknown chars with '?' to guarantee no crashes
        return text.encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Header ──
    pdf.set_fill_color(29, 78, 216)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "PharmaCare Drug Analysis Report", ln=1)
    pdf.set_font("Arial", size=9)
    pdf.set_xy(10, 19)
    pdf.cell(0, 6, "AI-Powered Medicine Fraud Detection System  |  v2.1", ln=1)

    # ── Original Image Section ──
    pdf.set_xy(10, 35)
    if image_bytes:
        tmpdir = tempfile.gettempdir()
        img_path = os.path.join(tmpdir, "pharmacheck_scan.jpg")
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, "Scanned Medicine", ln=1)
        pdf.image(img_path, x=10, w=90)
        pdf.ln(5)

    # ── Fraud Verdict Section ──
    if fraud_result:
        score, match, status = fraud_result
        is_fraud = score < FRAUD_THRESHOLD
        
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "Verdict", ln=1)
        
        # Verdict Box
        if is_fraud:
            pdf.set_fill_color(254, 226, 226) # light red
            pdf.set_text_color(220, 38, 38)
            verdict_text = "POTENTIALLY FRAUDULENT"
        else:
            pdf.set_fill_color(220, 252, 231) # light green
            pdf.set_text_color(22, 163, 74)
            verdict_text = "AUTHENTIC"
            
        pdf.set_font("Arial", "B", 12)
        pdf.cell(100, 10, f"{verdict_text} (Score: {score*100:.1f}%)", border=1, ln=1, fill=True)
        
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(15, 23, 42)
        pdf.multi_cell(0, 6, _sanitize(status))
        pdf.ln(8)

    # ── Extracted Data Table ──
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Extracted Named Entities", ln=1)
    
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(60, 8, "Entity Type", border=1, fill=True)
    pdf.cell(130, 8, "Detected Values", border=1, fill=True, ln=1)
    
    pdf.set_font("Arial", size=10)
    for label, values in detail.items():
        val_str = ", ".join(values)
        pdf.cell(60, 8, _sanitize(label), border=1)
        pdf.cell(130, 8, _sanitize(val_str), border=1, ln=1)

    # ── Footer ──
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100, 116, 139)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated by PharmaCare AI on {timestamp}  |  Page {pdf.page_no()}", align='C')

    tmpdir   = tempfile.gettempdir()
    pdf_path = os.path.join(tmpdir, "pharmacheck_summary.pdf")
    pdf.output(pdf_path)
    return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
#  Fraud Detection
# ─────────────────────────────────────────────────────────────────────────────

def fraud(detail: dict, raw_text: str = ""):
    """
    Compare extracted entities against the drug database using Jaccard similarity.
    If entities are sparse, falls back to raw OCR text for robustness.
    Returns (max_score: float, best_match: pd.Series, status: str).
    """
    df = load_and_preprocess_database()

    fields     = ["DRUGNAME", "TYPE", "COMPOSITION", "SIZE", "DOSAGE"]
    flat_words = [item for key in fields for item in (detail.get(key) or [])]

    # Robust Fallback for Live Demos: If NER missed the main keywords, use the raw text!
    text_to_analyze = ' '.join(map(str, flat_words))
    if len(text_to_analyze.split()) < 4 and raw_text:
        text_to_analyze = raw_text

    if not text_to_analyze.strip():
        return 0.0, pd.Series(dtype=object), "⚠️ Not enough entities extracted to evaluate."

    stop_words = set(stopwords.words("english"))
    punct      = {',', '.', ':', 'nan', '-', '_', '|'}
    input_set  = {
        t.lower() for t in word_tokenize(text_to_analyze)
        if t.lower() not in stop_words and t not in punct and len(t) > 2
    }

    if not input_set:
        return 0.0, pd.Series(dtype=object), "⚠️ No meaningful tokens found in entities."

    def _overlap(db_set):
        if not db_set: return 0.0
        intersection = len(db_set & input_set)
        denominator = min(len(db_set), len(input_set))
        sim = intersection / denominator if denominator else 0.0
        # Penalty for extremely short db records to prevent trivial matches
        if len(db_set) < 4:
            sim *= 0.8
        return sim

    scores     = df['token_set'].apply(_overlap)
    best_idx   = scores.idxmax()
    best_score = float(scores.max())

    display_cols = [c for c in df.columns if c != 'token_set']
    best_match   = df.loc[best_idx, display_cols]

    if best_score >= FRAUD_THRESHOLD:
        status = "This medicine appears authentic - it closely matches a known drug in the database."
    elif best_score >= 0.4:
        status = "Partial match found - verify this medicine manually with a pharmacist."
    else:
        status = "This medicine is potentially fraudulent - very low similarity to any known drug."

    return best_score, best_match, status
