import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime
import re

# ---------- Konfigurasi model ----------
MODEL_DIR = "JustinusNatanael/indobert-fake-review"
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
    # Asumsi: 0 = Real, 1 = Fake
    p_real, p_fake = float(probs[0]), float(probs[1])
    pred = "Fake" if p_fake >= p_real else "Real"
    conf = max(p_real, p_fake)
    return pred, p_real, p_fake, conf

def generate_reasoning(text, pred, p_real, p_fake):
    text_l = text.lower()
    reasons = []

    if pred == "Fake":
        if any(w in text_l for w in ["sangat puas", "puas sekali", "luar biasa", "perfect", "mantap sekali"]):
            reasons.append("Terlalu banyak kata positif yang berlebihan.")
        if re.search(r"\b(bagus){2,}\b", text_l) or re.search(r"\b(murah){2,}\b", text_l):
            reasons.append("Ada kata yang diulang-ulang seperti iklan.")
        if len(text.split()) < 8:
            reasons.append("Review sangat pendek sehingga terkesan tidak informatif.")
    else:  # Real
        if any(w in text_l for w in ["tapi", "namun", "minus", "kekurangan"]):
            reasons.append("Review berisi kelebihan dan kekurangan secara seimbang.")
        if 15 <= len(text.split()) <= 80:
            reasons.append("Panjang review wajar dan cukup detail.")
        if not re.search(r"(sangat|banget|sekali){2,}", text_l):
            reasons.append("Tidak ada pola pujian berulang yang berlebihan.")

    if not reasons:
        if pred == "Fake":
            reasons.append("Pola kalimat mirip promosi sehingga terdeteksi sebagai Fake.")
        else:
            reasons.append("Pola kalimat mirip review pengguna nyata sehingga terdeteksi sebagai Real.")

    return " ".join(reasons)

def badge(label: str):
    cls = "badge-fake" if label == "Fake" else "badge-real"
    return f'<span class="badge {cls}">{label}</span>'

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "helpful_count" not in st.session_state:
    st.session_state.helpful_count = 0

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## Pengaturan")

    threshold_pct = st.slider(
        "Batas minimal probabilitas Fake (%)",
        50.0, 95.0, 50.0, 1.0,
        help="Jika Prob Fake di atas persentase ini, review akan dianggap Fake."
    )
    threshold = threshold_pct / 100.0

    st.caption("Jika Prob Fake ≥ batas di atas, review diklasifikasikan sebagai Fake.")
    st.divider()
    st.markdown("### Tips input")
    st.write("- Tulis minimal 5–10 kata.")
    st.write("- Hindari teks kosong / cuma emoji.")
    st.divider()
    if st.button("Hapus histori"):
        st.session_state.history = []
        st.session_state.helpful_count = 0

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div>
        <div class="hero-title">Deteksi Fake Review (IndoBERT)</div>
        <div class="hero-subtitle">
        Masukkan review produk (Bahasa Indonesia) untuk melihat prediksi Real/Fake beserta tingkat keyakinan.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_history, tab_about = st.tabs(["Prediksi", "Histori", "Tentang"])

# ---------- Tab Prediksi ----------
with tab_pred:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Input review")
        text = st.text_area(
            "Masukkan review:",
            height=180,
            placeholder="Contoh: Barang cepat sampai, packing rapi, tapi kualitas kurang sesuai deskripsi..."
        )

        # Informasi tambahan
        st.markdown("#### Informasi tambahan (opsional)")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            rating = st.slider("Rating bintang", 1, 5, 5)
        with col_r2:
            helpful_count_input = st.number_input(
            "Berapa orang terbantu oleh review ini?",
            min_value=0,
            step=1,
            value=0,
            help="Contoh: di Tokopedia tertulis '12 orang merasa ulasan ini membantu'."
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

                pct_real = p_real * 100
                pct_fake = p_fake * 100
                pct_conf = conf * 100

                reason = generate_reasoning(text, pred, p_real, p_fake)

                st.markdown(
                    f"<div class='result-row'>Prediksi: {badge(pred)}</div>",
                    unsafe_allow_html=True
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Prob Real", f"{pct_real:.2f}%")
                m2.metric("Prob Fake", f"{pct_fake:.2f}%")
                m3.metric("Kepercayaan", f"{pct_conf:.2f}%")

                st.progress(min(conf, 1.0))

                st.markdown(f"**Reason:** {reason}")

                # update helpful counter
                if helpful:
                    st.session_state.helpful_count += 1

                # save history
                st.session_state.history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "pred": pred,
                        "p_real(%)": round(p_real * 100, 2),
                        "p_fake(%)": round(p_fake * 100, 2),
                        "rating": rating,
                        "helpful": helpful,
                        "text": text[:200] + ("..." if len(text) > 200 else ""),
                        "reason": reason,
                    }
                )

                if pred0 != pred:
                    st.info("Prediksi berubah karena batas probabilitas Fake yang kamu atur di kiri.")

        else:
            st.caption("Klik tombol Prediksi untuk melihat hasil.")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab History ----------
with tab_history:
    st.markdown("### Histori prediksi (session ini)")
    if len(st.session_state.history) == 0:
        st.caption("Belum ada histori.")
    else:
        st.markdown(
            f"Total review yang ditandai **membantu** di sesi ini: {st.session_state.helpful_count}"
        )
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Tab About ----------
with tab_about:
    st.markdown("### Tentang aplikasi")
    st.write(
        "Aplikasi ini memuat model IndoBERT hasil fine-tuning dan melakukan inference di CPU/GPU untuk klasifikasi Fake/Real."
    )
    st.write(
        "Tampilan menggunakan theming (config.toml) dan CSS sederhana untuk memberikan pengalaman pengguna yang lebih nyaman."
    )

