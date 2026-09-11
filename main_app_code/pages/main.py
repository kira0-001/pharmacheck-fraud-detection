# ─────────────────────────────────────────────────────────────────────────────
#  PharmaCare · pages/main.py  —  Full UI  v2.0
#  Entry point: streamlit run main_app_code\pages\main.py
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import time
import base64
import cv2
import numpy as np
import sys
import os
from PIL import Image

# Add the main_app_code directory to sys.path so ner_app can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Path Setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..")))

from config import *          # APP_NAME, APP_ICON, FRAUD_THRESHOLD, etc.

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="spacy")

# ── Lazy import ner_app so the UI renders instantly before heavy models load ──
@st.cache_resource(show_spinner=False)
def _get_ner_app():
    import ner_app as _ner
    return _ner

def _ner():
    """Convenience accessor so we can write _ner().function() everywhere."""
    return _get_ner_app()

# ── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} | Fraud Detection",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stMarkdown, .stText  {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Hide Streamlit chrome safely ── */
header { visibility: hidden !important; }
[data-testid="stHeader"] { visibility: hidden !important; }
[data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(29,78,216,0.4); border-radius: 9px; }
::-webkit-scrollbar-thumb:hover { background: #1d4ed8; }

.main .block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 4rem !important;
    max-width: 1100px !important;
}

/* ─── ANIMATIONS ──────────────────────────────────────────────────────────── */
@keyframes fadeUp  { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
@keyframes popIn   { 0% { transform:scale(0); } 80% { transform:scale(1.15); } 100% { transform:scale(1); } }
@keyframes pulse   { 0%,100% { box-shadow: 0 0 0 0 rgba(29,78,216,0.4); } 50% { box-shadow: 0 0 0 10px rgba(29,78,216,0); } }
@keyframes shimmer { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
@keyframes float   { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-6px); } }

.pc-card, .hero, .steps-bar, .ocr-box, .verdict-ok, .verdict-fraud { animation: fadeUp 0.55s cubic-bezier(0.16,1,0.3,1) forwards; }

/* ─── HERO ────────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #060d24 0%, #0c1e55 35%, #1d4ed8 75%, #0c5fa8 100%);
    padding: 2.8rem 3rem;
    border-radius: 24px;
    color: white;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(13, 36, 96, 0.6), inset 0 1px 0 rgba(255,255,255,0.08);
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}
.hero:hover { transform: translateY(-3px); box-shadow: 0 28px 72px rgba(13,36,96,0.7), inset 0 1px 0 rgba(255,255,255,0.1); }
/* Animated glowing orbs */
.hero::before {
    content: '';
    position: absolute;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(59,130,246,0.35) 0%, transparent 70%);
    top: -80px; right: 60px;
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}
.hero::after {
    content: '💊';
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.08;
    animation: float 4s ease-in-out infinite;
    filter: blur(1px);
}
.hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px; padding: 5px 16px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 1.1rem;
    color: rgba(255,255,255,0.88);
    backdrop-filter: blur(8px);
}
.hero h1 { font-size: 2.6rem; font-weight: 900; margin: 0 0 0.5rem 0; letter-spacing: -1px; line-height: 1.1; }
.hero p { font-size: 0.95rem; opacity: 0.72; margin: 0; font-weight: 400; max-width: 560px; line-height: 1.75; }

/* ─── STEP BAR ────────────────────────────────────────────────────────────── */
.steps-bar {
    display: flex; align-items: flex-start;
    background: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    border-radius: 16px; padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 6px 24px rgba(0,0,0,0.09);
}
.step-wrap { display: flex; flex-direction: column; align-items: center; flex: 1; gap: 0.45rem; position: relative; }
.step-circle {
    width: 40px; height: 40px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.85rem; border: 2.5px solid;
    transition: all 0.45s cubic-bezier(0.34, 1.56, 0.64, 1); z-index: 2;
}
.step-circle.done   { background: linear-gradient(135deg,#059669,#047857); border-color: #059669; color: white; transform: scale(1.08); }
.step-circle.active {
    background: linear-gradient(135deg,#1d4ed8,#2563eb); border-color: #1d4ed8; color: white;
    box-shadow: 0 0 0 6px rgba(29,78,216,0.2); transform: scale(1.15);
    animation: pulse 2s infinite;
}
.step-circle.todo   { background: var(--secondary-background-color); border-color: var(--text-color); opacity: 0.25; color: var(--text-color); }
.step-lbl           { font-size: 0.72rem; font-weight: 600; color: var(--text-color); opacity: 0.55; text-align: center; line-height: 1.3; transition: all 0.3s; }
.step-lbl.active    { color: #1d4ed8; opacity: 1; font-weight: 700; }
.step-lbl.done      { color: #059669; opacity: 1; }
.step-sep { flex: 1; height: 3px; margin-top: 20px; border-radius: 3px; max-width: 80px; transition: background 0.5s; }
.step-sep.done { background: linear-gradient(90deg, #059669, #34d399); }
.step-sep.todo { background: var(--secondary-background-color); opacity: 0.4; }

/* ─── SECTION CARD ─────────────────────────────────────────────────────────── */
.pc-card {
    background: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    border-left: 4px solid #1d4ed8;
    border-radius: 14px; padding: 1.6rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s;
}
.pc-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.13); border-left-color: #2563eb; }
.pc-card-title {
    font-size: 1.02rem; font-weight: 700; color: var(--text-color);
    margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;
    padding-bottom: 0.75rem; border-bottom: 1px solid var(--secondary-background-color);
}

/* ─── OCR BOX ────────────────────────────────────────────────────────────── */
.ocr-box {
    background: var(--secondary-background-color);
    border: 1px solid var(--secondary-background-color);
    border-left: 4px solid #1d4ed8;
    border-radius: 8px; padding: 1rem 1.25rem;
    font-size: 0.86rem; line-height: 2.0; color: var(--text-color);
    max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-word;
    font-family: 'Inter', monospace;
}

/* ─── SCORE / VERDICT ────────────────────────────────────────────────────────── */
.verdict-ok {
    background: linear-gradient(135deg, rgba(5,150,105,0.12), rgba(16,185,129,0.06));
    border: 2px solid #059669;
    border-radius: 18px; padding: 2.2rem; text-align: center;
    box-shadow: 0 0 40px rgba(5,150,105,0.12);
}
.verdict-fraud {
    background: linear-gradient(135deg, rgba(220,38,38,0.12), rgba(239,68,68,0.06));
    border: 2px solid #dc2626;
    border-radius: 18px; padding: 2.2rem; text-align: center;
    box-shadow: 0 0 40px rgba(220,38,38,0.12);
}
.v-icon   { font-size: 3.2rem; display: block; margin-bottom: 0.5rem; line-height: 1; animation: popIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
.v-title  { font-size: 1.6rem; font-weight: 900; margin: 0; letter-spacing: -0.4px; }
.v-sub    { font-size: 0.88rem; opacity: 0.75; margin: 0.5rem 0 0 0; }
.v-score  { font-size: 4rem; font-weight: 900; display: block; line-height: 1.1; letter-spacing: -2px; }
.sbar     { background: var(--secondary-background-color); border-radius: 999px; height: 14px; overflow: hidden; margin: 0.6rem 0; }
.sbar-f   { height: 100%; border-radius: 999px; transition: width 1.8s cubic-bezier(0.16, 1, 0.3, 1); }

/* ─── SIDEBAR ────────────────────────────────────────────────────────────── */
/* Nuke the running man & status widget in every known selector */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] *,
.stStatusWidget,
.reportview-container .stDecoration { display: none !important; visibility: hidden !important; }
header .stToolbar [data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stSidebar"] {
    border-right: 1px solid var(--secondary-background-color) !important;
}
.sb-brand {
    background: linear-gradient(160deg, #060d24 0%, #0f2460 50%, #1d4ed8 100%);
    border-radius: 14px; padding: 1.4rem 1.1rem;
    text-align: center; margin-bottom: 1.25rem; color: white;
    box-shadow: 0 8px 24px rgba(13,36,96,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
}
.sb-brand-icon { font-size: 2.8rem; line-height: 1; }
.sb-brand-name { font-size: 1.2rem; font-weight: 800; letter-spacing: -0.4px; margin-top: 0.35rem; }
.sb-brand-ver  { font-size: 0.6rem; opacity: 0.6; margin-top: 3px; letter-spacing: 0.5px; }
.sb-section {
    background: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    border-radius: 10px; padding: 0.9rem 1rem; margin-bottom: 0.75rem;
}
.sb-sec-title {
    font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text-color); opacity: 0.5; margin-bottom: 0.65rem;
}
.sb-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; padding: 0.28rem 0; color: var(--text-color); }
.sb-val { font-weight: 700; color: #3b82f6; }

/* ─── BUTTONS ─────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: none !important; letter-spacing: 0.2px !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(0,0,0,0.18) !important; }
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stFileUploader"] > section {
    border: 2px dashed rgba(29,78,216,0.4) !important;
    border-radius: 14px !important;
    background: rgba(29,78,216,0.03) !important;
    transition: all 0.25s !important;
}
[data-testid="stFileUploader"] > section:hover { border-color: #1d4ed8 !important; background: rgba(29,78,216,0.07) !important; }

/* ─── MISC ─────────────────────────────────────────────────────────────────── */
.tip-box {
    background: rgba(29,78,216,0.07); border: 1px solid rgba(29,78,216,0.2);
    border-left: 4px solid #1d4ed8; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.85rem; color: var(--text-color);
    margin-top: 0.75rem;
}
</style>

<script>
(function() {
  // Kill the Streamlit running-man animation via JavaScript
  // CSS alone can't reliably target it in all Streamlit versions
  function killRunner() {
    var selectors = [
      '[data-testid="stStatusWidget"]',
      '[data-testid="stAppStatus"]',
      '.stStatusWidget',
      '.stDecoration'
    ];
    selectors.forEach(function(sel) {
      document.querySelectorAll(sel).forEach(function(el) {
        el.style.display = 'none';
        el.style.visibility = 'hidden';
      });
    });
  }
  // Run immediately
  killRunner();
  // Also watch for dynamic DOM changes
  var obs = new MutationObserver(killRunner);
  obs.observe(document.documentElement, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "step":             1,
    "opencv_image":     None,
    "contour_bytes":    None,
    "extracted_text":   "",
    "ner_html":         "",
    "details_dict":     None,
    "pdf_bytes":        None,
    "fraud_results":    None,
    "uploader_key":     0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _advance():  st.session_state.step = min(4, st.session_state.step + 1)
def _back():     st.session_state.step = max(1, st.session_state.step - 1)

def _reset():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.uploader_key += 1


# ─────────────────────────────────────────────────────────────────────────────
#  Components
# ─────────────────────────────────────────────────────────────────────────────
def _hero():
    st.markdown(f"""
<div class="hero">
    <div class="hero-pill">🔬 AI-Powered · v{APP_VERSION}</div>
    <h1>{APP_ICON} {APP_NAME}</h1>
    <p>{APP_SUBTITLE} — Upload a medicine package image to instantly verify its authenticity using OCR, custom NLP, and AI similarity scoring.</p>
</div>""", unsafe_allow_html=True)


def _step_bar(current: int):
    labels = ["Upload Image", "Extract Text", "NER Analysis", "Fraud Check"]
    icons  = ["📷",           "📝",           "🧬",           "🛡️"]
    html   = '<div class="steps-bar">'
    for i, (icon, label) in enumerate(zip(icons, labels), 1):
        s = "done" if i < current else ("active" if i == current else "todo")
        circle_inner = "✓" if s == "done" else str(i)
        html += (
            f'<div class="step-wrap">'
            f'  <div class="step-circle {s}">{circle_inner}</div>'
            f'  <div class="step-lbl {s}">{icon}&nbsp;{label}</div>'
            f'</div>'
        )
        if i < 4:
            sep_s = "done" if i < current else "todo"
            html += f'<div class="step-sep {sep_s}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Language Mapping ─────────────────────────────────────────────────────────
LANGUAGE_MAP = {
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu"
}

def _sidebar():
    with st.sidebar:
        st.session_state.app_mode = st.radio("Mode", ["Live Demo", "Database Admin"], horizontal=True)
        st.markdown(f"""
<div class="sb-brand">
    <div class="sb-brand-icon">{APP_ICON}</div>
    <div class="sb-brand-name">{APP_NAME}</div>
    <div class="sb-brand-ver">v{APP_VERSION} · Fraud Detection Engine</div>
</div>""", unsafe_allow_html=True)

        st.session_state.selected_lang = st.selectbox(
            "Scan Language",
            options=list(LANGUAGE_MAP.keys()),
            index=0
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Pipeline status
        pipeline = [
            ("📷 Image Loaded",  st.session_state.opencv_image is not None),
            ("📝 Text Extracted", bool(st.session_state.extracted_text)),
            ("🧬 NER Complete",   st.session_state.details_dict is not None),
            ("🛡️ Fraud Checked",  st.session_state.fraud_results is not None),
        ]
        rows_html = "".join(
            f'<div class="sb-row"><span>{name}</span>'
            f'<span>{"✅" if done else "⬜"}</span></div>'
            for name, done in pipeline
        )
        st.markdown(f'<div class="sb-section"><div class="sb-sec-title">Pipeline Status</div>{rows_html}</div>',
                    unsafe_allow_html=True)

        # DB stats - read from config, don't eagerly load the database
        try:
            db = _ner().load_and_preprocess_database()
            n_drugs = f"{len(db):,}"
        except Exception:
            n_drugs = "Loading..."

        threshold_label = f"{int(FRAUD_THRESHOLD * 100)}% Jaccard"
        st.markdown(f"""
<div class="sb-section">
    <div class="sb-sec-title">System Info</div>
    <div class="sb-row"><span>🗄️ Drug Records</span><span class="sb-val">{n_drugs}</span></div>
    <div class="sb-row"><span>🤖 NER Model</span><span class="sb-val">Active ✓</span></div>
    <div class="sb-row"><span>📏 Threshold</span><span class="sb-val">{threshold_label}</span></div>
    <div class="sb-row"><span>🔍 Engine</span><span class="sb-val">Tesseract OCR</span></div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🔄 New Scan", width="stretch"):
            _reset(); st.rerun()

        if st.session_state.pdf_bytes:
            st.download_button(
                label="📄 Download PDF Report",
                data=st.session_state.pdf_bytes,
                file_name="pharmacheck_report.pdf",
                mime="application/pdf",
                width="stretch",
            )

        st.markdown("""
<div style="font-size:0.72rem;color:#94a3b8;line-height:1.8;margin-top:0.75rem;">
<b>How it works</b><br>
① Upload image<br>
② Tesseract OCR extracts text<br>
③ spaCy NER tags entities<br>
④ Jaccard similarity checks authenticity against drug database
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — Image Upload
# ─────────────────────────────────────────────────────────────────────────────
def _step1():
    st.markdown("### 📷 Step 1 — Upload Medicine Image")
    st.markdown("Upload a clear photo of a **medicine box, label, or package**. Supported: JPG, PNG, BMP, TIFF, WEBP.")

    col_upload, col_preview = st.columns([3, 2], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Drop your image here or click Browse",
            type=SUPPORTED_FORMATS,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )

        if uploaded:
            size_mb = uploaded.size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File is too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB.")
                return

            raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None:
                st.error("❌ Could not decode image. Please try a different file.")
                return

            st.session_state.opencv_image  = img
            st.session_state.extracted_text = ""          # reset downstream state
            st.session_state.ner_html       = ""
            st.session_state.details_dict   = None
            st.session_state.fraud_results  = None
            st.session_state.contour_bytes  = None
            st.session_state.pdf_bytes      = None

        st.markdown("""
<div class="tip-box">
💡 <b>Tip:</b> For best results use a well-lit, flat photo.
The AI works on any printed or packaged medicine label.
</div>""", unsafe_allow_html=True)

    with col_preview:
        if st.session_state.opencv_image is not None:
            h, w = st.session_state.opencv_image.shape[:2]
            st.image(st.session_state.opencv_image, channels="BGR",
                     caption=f"Preview — {w}×{h}px", width="stretch")
            st.success("✅ Image loaded and ready.")

    if st.session_state.opencv_image is not None:
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            if st.button("Extract Text  →", type="primary", width="stretch"):
                _advance(); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — OCR Extraction
# ─────────────────────────────────────────────────────────────────────────────
def _step2():
    st.markdown("### 📝 Step 2 — OCR Text Extraction")

    # Run OCR if not already done
    if not st.session_state.extracted_text:
        lang_code = LANGUAGE_MAP.get(st.session_state.selected_lang, "eng")
        with st.spinner("🔍 Running OCR… please wait."):
            try:
                text, result = _ner().ocr_extraction(st.session_state.opencv_image, lang_code=lang_code)
            except Exception as e:
                st.error(f"❌ OCR failed: {e}"); return

        if len(text.strip()) < MIN_TEXT_LENGTH:
            st.warning(
                f"⚠️ Very little text detected ({len(text.strip())} characters). "
                "Try a clearer or higher-resolution photo."
            )
            if st.button("← Try a different image"):
                _back(); st.rerun()
            return

        st.session_state.extracted_text = text

        # Store contour image as bytes to avoid large numpy array in state
        img_copy    = st.session_state.opencv_image.copy()
        contour_img = _ner().draw_contours(img_copy, result)
        _, buf = cv2.imencode('.jpg', contour_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        st.session_state.contour_bytes = buf.tobytes()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="pc-card"><div class="pc-card-title">🖼️ Original Image</div>', unsafe_allow_html=True)
        st.image(st.session_state.opencv_image, channels="BGR", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="pc-card"><div class="pc-card-title">🟩 Detected Text Regions</div>', unsafe_allow_html=True)
        if st.session_state.contour_bytes:
            st.image(st.session_state.contour_bytes, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("**Extracted Text:**")
    st.markdown(
        f'<div class="ocr-box">{st.session_state.extracted_text}</div>',
        unsafe_allow_html=True
    )

    # Quick metrics
    words = len(st.session_state.extracted_text.split())
    chars = len(st.session_state.extracted_text)
    m1, m2, m3 = st.columns(3)
    m1.metric("Words Detected",  words)
    m2.metric("Characters",      chars)
    m3.metric("OCR Engine",      "Tesseract")

    nav1, nav2, _ = st.columns([1, 1, 3])
    with nav1:
        if st.button("← Back",         width="stretch"): _back();    st.rerun()
    with nav2:
        if st.button("Analyze NER  →", width="stretch", type="primary"): _advance(); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — NER Analysis
# ─────────────────────────────────────────────────────────────────────────────
def _step3():
    st.markdown("### 🧬 Step 3 — Named Entity Recognition")

    if st.session_state.details_dict is None:
        with st.spinner("🤖 Running AI entity recognition on extracted text…"):
            try:
                doc = _ner().perform_named_entity_recognition(st.session_state.extracted_text)
                st.session_state.ner_html    = _ner().display_doc(doc)
                st.session_state.details_dict = _ner().details_dict(doc)
                # We generate the PDF in step 4 now when we have the verdict
            except Exception as e:
                st.error(f"❌ NER failed: {e}"); return

    # Entity displacy render
    if st.session_state.ner_html:
        st.markdown('<div class="pc-card"><div class="pc-card-title">🔖 AI-Tagged Entities</div>', unsafe_allow_html=True)
        st.markdown(st.session_state.ner_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Entity table
    details = st.session_state.details_dict or {}
    if details:
        rows = [{"Entity Type": lbl, "Extracted Values": ", ".join(vals)}
                for lbl, vals in details.items()]
        st.markdown('<div class="pc-card"><div class="pc-card-title">📋 Entity Summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No named entities were recognised. The model may not have matched any known pharmaceutical terms in the extracted text.")

    key_entities = {"DRUGNAME", "COMPOSITION", "TYPE"}
    has_keys = bool(key_entities & set(details.keys()))
    if not has_keys and details:
        st.info("ℹ️ Key entities (DRUGNAME, COMPOSITION, TYPE) not found. Fraud detection accuracy will be limited.")

    nav1, nav2, _ = st.columns([1, 1, 3])
    with nav1:
        if st.button("← Back",            width="stretch"): _back();    st.rerun()
    with nav2:
        if st.button("Check Fraud  →",    width="stretch", type="primary"): _advance(); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 — Fraud Detection
# ─────────────────────────────────────────────────────────────────────────────
def _step4():
    st.markdown("### 🛡️ Step 4 — Fraud Detection Result")

    if st.session_state.fraud_results is None:
        with st.spinner("🔬 Comparing against drug database… please wait."):
            try:
                score, match, status = _ner().fraud(
                    st.session_state.details_dict, 
                    raw_text=st.session_state.extracted_text
                )
                st.session_state.fraud_results = (score, match, status)
                
                # Generate PDF with complete results
                img_bytes = None
                if st.session_state.opencv_image is not None:
                    _, buf = cv2.imencode('.jpg', st.session_state.opencv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_bytes = buf.tobytes()
                    
                pdf_path = _ner().create_summary_pdf(
                    st.session_state.details_dict, 
                    image_bytes=img_bytes,
                    fraud_result=st.session_state.fraud_results
                )
                with open(pdf_path, "rb") as f:
                    st.session_state.pdf_bytes = f.read()
            except Exception as e:
                st.error(f"❌ Fraud detection failed: {e}"); return

    score, match, status = st.session_state.fraud_results
    is_fraud    = score < FRAUD_THRESHOLD
    score_pct   = round(score * 100, 1)
    bar_color   = "#ef4444" if is_fraud else "#22c55e"
    verdict_cls = "verdict-fraud" if is_fraud else "verdict-ok"
    verdict_icon  = "🚨" if is_fraud else "✅"
    verdict_label = "POTENTIALLY FRAUDULENT" if is_fraud else "AUTHENTIC"
    v_color     = "#dc2626" if is_fraud else "#15803d"

    col_score, col_verdict = st.columns(2, gap="large")

    with col_score:
        thr_pct = int(FRAUD_THRESHOLD * 100)
        st.markdown(f"""
<div class="pc-card">
  <div class="pc-card-title">📊 Jaccard Similarity Score</div>
  <div style="text-align:center; padding:0.5rem 0;">
    <span class="v-score" style="color:{bar_color};">{score_pct}%</span>
    <div style="font-size:0.78rem; color:#64748b; margin:0.2rem 0;">
        Best match against {'{:,}'.format(len(_ner().load_and_preprocess_database()))} drug records
    </div>
  </div>
  <div class="sbar">
    <div class="sbar-f" style="width:{score_pct}%; background:{bar_color};"></div>
  </div>
  <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:#94a3b8;">
    <span>0%</span><span>Threshold: {thr_pct}%</span><span>100%</span>
  </div>
</div>""", unsafe_allow_html=True)

    with col_verdict:
        st.markdown(f"""
<div class="{verdict_cls}">
  <span class="v-icon">{verdict_icon}</span>
  <p class="v-title" style="color:{v_color};">{verdict_label}</p>
  <p class="v-sub">{status}</p>
</div>""", unsafe_allow_html=True)

    # Best match table
    st.markdown('<div class="pc-card"><div class="pc-card-title">🗄️ Best Matched Database Record</div>', unsafe_allow_html=True)
    if hasattr(match, 'to_dict'):
        skip = {'IMAGE_PATH', 'Folder_name', 'token_set'}
        show = {k: v for k, v in match.to_dict().items() if k not in skip}
        if show:
            tbl = pd.DataFrame([show])
            st.dataframe(tbl, width="stretch", hide_index=True)
        else:
            st.info("No displayable fields in the matched record.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Interpretation
    with st.expander("ℹ️ How to interpret this result"):
        thr_pct = int(FRAUD_THRESHOLD * 100)
        drug_name = match.get('DRUGNAME', 'Unknown') if hasattr(match, 'get') else "Unknown"
        st.markdown(f"""
| Score | Interpretation |
|---|---|
| ≥ {thr_pct}% | ✅ Likely **authentic** — strong match in database |
| < {thr_pct}% | 🚨 **Potentially fraudulent** — weak or no match |

- **Your score:** `{score_pct}%`
- **Best matching drug:** `{drug_name}`
- The score is computed as Jaccard similarity between the NER entities from your image and every record in the drug database.
- A score of `0%` means zero shared tokens. A score of `100%` means an exact match.
        """)

    nav1, _, col_restart = st.columns([1, 2, 1])
    with nav1:
        if st.button("← Back",       width="stretch"): _back(); st.rerun()
    with col_restart:
        if st.button("🔄 New Scan",   width="stretch"): _reset(); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Admin Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def _admin_dashboard():
    n = _ner  # alias to avoid calling it repeatedly
    db_path = DATA_XLSX

    try:
        df = pd.read_excel(db_path)
    except Exception as e:
        st.error(f"❌ Could not load database: {e}")
        return

    total   = len(df)
    unique  = df['DRUGNAME'].nunique() if 'DRUGNAME' in df.columns else 0
    types   = df['TYPE'].nunique()     if 'TYPE'     in df.columns else 0

    st.markdown("""
<div style="background:linear-gradient(135deg,#060d24,#0f2460,#1d4ed8);
            border-radius:18px;padding:2rem 2.5rem;color:white;margin-bottom:1.5rem;
            box-shadow:0 12px 40px rgba(13,36,96,0.5);">
  <div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
              opacity:0.6;margin-bottom:0.5rem;">DEVELOPER PANEL</div>
  <h2 style="margin:0;font-size:1.8rem;font-weight:900;letter-spacing:-0.5px;">🗄️ Database Manager</h2>
  <p style="margin:0.4rem 0 0 0;opacity:0.7;font-size:0.9rem;">View, search, and add medicines to the fraud detection database.</p>
</div>""", unsafe_allow_html=True)

    # Stats row
    c1, c2, c3 = st.columns(3)
    for col, icon, label, value in [
        (c1, "🗄️", "Total Records",   total),
        (c2, "💊", "Unique Drugs",    unique),
        (c3, "🏷️", "Drug Categories", types),
    ]:
        col.markdown(f"""
<div style="background:var(--background-color);border:1px solid var(--secondary-background-color);
            border-left:4px solid #1d4ed8;border-radius:12px;padding:1.2rem 1.4rem;text-align:center;">
  <div style="font-size:1.8rem;">{icon}</div>
  <div style="font-size:2rem;font-weight:900;color:#3b82f6;line-height:1.1;">{value:,}</div>
  <div style="font-size:0.78rem;opacity:0.6;margin-top:2px;">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Search + Table
    search = st.text_input("🔍 Search medicines", placeholder="Type a drug name or composition...")
    if search:
        mask = df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
        filtered = df[mask]
    else:
        filtered = df

    st.markdown(f"Showing **{len(filtered):,}** of **{total:,}** records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Add form
    st.markdown("""
<div style="background:var(--background-color);border:1px solid var(--secondary-background-color);
            border-left:4px solid #059669;border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1rem;">
  <div style="font-size:1.05rem;font-weight:700;margin-bottom:1rem;">➕ Add New Medicine to Database</div>""",
        unsafe_allow_html=True)

    with st.form("add_drug_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        drugname    = col1.text_input("💊 Drug Name", placeholder="e.g. PENTAB-20")
        drugtype    = col2.text_input("🏷️ Type", placeholder="e.g. Tablets")
        composition = st.text_area("🧪 Composition",
                                   placeholder="e.g. Pantoprazole Sodium Sesquihydrate 20mg",
                                   height=90)
        col3, col4 = st.columns(2)
        size   = col3.text_input("📍 Size",   placeholder="e.g. 20 mg")
        dosage = col4.text_input("⏱️ Dosage", placeholder="e.g. As directed by physician")

        submitted = st.form_submit_button("➕ Add to Database", type="primary", use_container_width=True)
        if submitted:
            if drugname and composition:
                new_row = {"DRUGNAME": drugname, "TYPE": drugtype,
                           "COMPOSITION": composition, "SIZE": size, "DOSAGE": dosage}
                updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                updated_df.to_excel(db_path, index=False)
                st.cache_resource.clear()
                st.success(f"✅ **{drugname}** added! Database now has {len(updated_df):,} records.")
                st.rerun()
            else:
                st.error("❌ Drug Name and Composition are required!")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
_inject_css()
_sidebar()

if st.session_state.get("app_mode") == "Database Admin":
    _admin_dashboard()
else:
    _hero()
    _step_bar(st.session_state.step)
    {1: _step1, 2: _step2, 3: _step3, 4: _step4}[st.session_state.step]()