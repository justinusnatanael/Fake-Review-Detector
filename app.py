import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime

MODEL_DIR = "indobert_fake_review_model"
MAX_LEN = 128

# ---------- Page config ----------
st.set_page_config(
    page_title="Fake Review Detector (IndoBERT)",
    page_icon="🧠",
    layout="wide",
)

# ---------- Load CSS ----------
def load_css(path: str):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Buat file assets/style.css (contoh ada di bawah)
load_css("assets/style.css")

# ---------- Model loader ----------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------- Helpers ----------
def predict(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # [p0, p1]
    # Asumsi: label 0=Real, 1=Fake (sesuai app kamu sebelumnya)
    p_real, p_fake = float(probs[0]), float(probs[1])
    pred = "Fake" if p_fake >= p_real else "Real"
    conf = max(p_real, p_fake)
    return pred, p_real, p_fake, conf

def badge(label: str):
    cls = "badge-fake" if label == "Fake" else "badge-real"
    return f'<span class="badge {cls}">{label}</span>'

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## Pengaturan")
    threshold = st.slider("Ambang Fake (threshold p_fake)", 0.50, 0.95, 0.50, 0.01)
    st.caption("Jika p_fake ≥ threshold → Fake.")
    st.divider()
    st.markdown("### Tips input")
    st.write("- Tulis minimal 5–10 kata.")
    st.write("- Hindari teks kosong / cuma emoji.")
    st.divider()
    if st.button("Hapus histori"):
        st.session_state.history = []

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div>
        <div class="hero-title">Deteksi Fake Review (IndoBERT)</div>
        <div class="hero-subtitle">Masukkan review produk (Bahasa Indonesia) untuk melihat prediksi Real/Fake beserta confidence.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_history, tab_about = st.tabs(["Prediksi", "Histori", "Tentang"])

with tab_pred:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Input review")
        text = st.text_area(
            "Masukkan review:",
            height=180,
            placeholder="Contoh: Barang cepat sampai, packing rapi, tapi kualitas kurang sesuai deskripsi..."
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            do_pred = st.button("Prediksi", use_container_width=True)
        with c2:
            st.button("Reset", use_container_width=True, on_click=lambda: None)

    with right:
        st.markdown("### Hasil")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if do_pred:
            if not text.strip():
                st.warning("Teks masih kosong.")
            else:
                pred0, p_real, p_fake, conf = predict(text)

                # apply threshold from sidebar
                pred = "Fake" if p_fake >= threshold else "Real"

                st.markdown(
                    f"<div class='result-row'>Prediksi: {badge(pred)}</div>",
                    unsafe_allow_html=True
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Prob Real", f"{p_real:.4f}")
                m2.metric("Prob Fake", f"{p_fake:.4f}")
                m3.metric("Confidence", f"{conf:.4f}")

                st.progress(min(conf, 1.0))

                # save history
                st.session_state.history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "pred": pred,
                        "p_real": round(p_real, 4),
                        "p_fake": round(p_fake, 4),
                        "text": text[:200] + ("..." if len(text) > 200 else "")
                    }
                )

                if pred0 != pred:
                    st.info("Prediksi berubah karena threshold (ambang) di sidebar.")

        else:
            st.caption("Klik tombol Prediksi untuk melihat hasil.")

        st.markdown("</div>", unsafe_allow_html=True)

with tab_history:
    st.markdown("### Histori prediksi (session ini)")
    if len(st.session_state.history) == 0:
        st.caption("Belum ada histori.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab_about:
    st.markdown("### Tentang aplikasi")
    st.write(
        "Aplikasi ini memuat model IndoBERT hasil fine-tuning dan melakukan inference di CPU/GPU untuk klasifikasi Fake/Real."
    )
    st.write("Untuk tampilan, app memakai theming (config.toml) dan CSS sederhana. [web:21][web:33]")

