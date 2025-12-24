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

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div>
        <div class="hero-title">Deteksi Fake Review </div>
        <div class="hero-subtitle">
        Masukkan review produk (Bahasa Indonesia) untuk melihat prediksi Real/Fake beserta tingkat keyakinan.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs: Prediksi, Histori, Indikator, Tentang
tab_pred, tab_history, tab_rules, tab_about = st.tabs(
    ["Prediksi", "Histori", "Indikator Fake Review", "Tentang"]
)

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
                st.markdown(
                    f"**Info tambahan:** {helpful_count_input} orang merasa review ini membantu (input pengguna)."
                )

                # save history
                st.session_state.history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "pred": pred,
                        "p_real(%)": round(p_real * 100, 2),
                        "p_fake(%)": round(p_fake * 100, 2),
                        "rating": rating,
                        "helpful_count": int(helpful_count_input),
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
        st.caption(
            "Kolom 'helpful_count' menunjukkan berapa orang yang menurut pengguna terbantu "
            "oleh review tersebut (mirip informasi jempol di e-commerce)."
        )
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Tab Indikator Fake Review ----------
with tab_rules:
    st.markdown("### Indikator Fake Review (Ringkas)")
    st.write(
        "Bagian ini merangkum ciri-ciri umum ulasan yang cenderung Fake dan yang cenderung Real, "
        "berdasarkan aturan linguistik dan pola isi review."
    )

    st.markdown("#### 1. Emosi & Sentimen Bahasa")
    st.markdown(
        "- **Cenderung Fake**: Sentimen ekstrem (terlalu positif / terlalu negatif) tanpa alasan jelas, "
        "banyak kata hiperbola seperti *luar biasa, terbaik di dunia, parah banget, hancur total*.\n"
        "- **Cenderung Real**: Sentimen lebih seimbang, biasanya ada kelebihan **dan** kekurangan sekaligus."
    )
    st.markdown("**Contoh Fake:** *\"Produk ini SEMPURNA banget!!! Terbaik di dunia, wajib beli pokoknya!!!\"*")
    st.markdown("**Contoh Real:** *\"Secara umum bagus, bahan cukup tebal dan nyaman, tapi jahitannya di bagian "
                "lengan agak kurang rapi.\"*")

    st.markdown("#### 2. Emoji & Tanda Baca")
    st.markdown(
        "- **Cenderung Fake**: Emoji berlebihan atau berulang (😊😊😊🔥🔥🔥💯💯💯), tanda seru/kata kapital berlebihan "
        "seperti *\"PRODUK INI WAJIB BELI!!!\"*.\n"
        "- **Cenderung Real**: Emoji sedikit atau tidak ada, tanda baca dan huruf kapital digunakan secara wajar."
    )
    st.markdown("**Contoh Fake:** *\"Keren banget barangnya 😊😊😊🔥🔥🔥💯💯💯\"*")
    st.markdown("**Contoh Real:** *\"Packing rapi, pengiriman cepat 🙂 barang sesuai foto.\"*")

    st.markdown("#### 3. Panjang & Struktur Kalimat")
    st.markdown(
        "- **Cenderung Fake**: Review sangat pendek dan generik (*\"Bagus banget\", \"Recommended\"*) atau sangat "
        "panjang tapi isinya seperti promosi dan tidak fokus pada pengalaman nyata.\n"
        "- **Cenderung Real**: Panjang wajar, kalimat mengalir alami dan fokus pada pengalaman pribadi."
    )
    st.markdown("**Contoh Fake (pendek):** *\"Recommended banget, pokoknya mantap\"*")
    st.markdown("**Contoh Real:** *\"Ukuran L pas di badan (tinggi 170cm, 65kg). Setelah dicuci 2x warnanya belum pudar, "
                "tapi resleting agak seret.\"*")

    st.markdown("#### 4. Detail Konten & Bahasa Promosi")
    st.markdown(
        "- **Cenderung Fake**: Jarang menyebut detail produk (warna, ukuran, kondisi pemakaian), banyak kata "
        "marketing seperti *wajib beli, best seller, dijamin puas*.\n"
        "- **Cenderung Real**: Menyebut detail konkret (pengiriman, kualitas material, daya tahan) dengan bahasa "
        "sehari-hari yang tidak terkesan iklan."
    )
    st.markdown("**Contoh Fake:** *\"Best seller banget, kualitas premium, dijamin puas pokoknya!\"*")
    st.markdown("**Contoh Real:** *\"Kardus sedikit penyok tapi isi aman. Suara kipas cukup halus, dipakai 3 jam terus "
                "belum panas. Kabelnya agak pendek jadi perlu colokan dekat meja.\"*")

    st.markdown("#### 5. Contoh output model IndoBERT")
    st.write(
        "Berikut dua contoh bagaimana indikator di atas tercermin pada probabilitas model:"
    )
    st.markdown(
        "- **Contoh Real**: *\"Kardus sedikit penyok tapi isi aman. Suara kipas cukup halus, dipakai 3 jam terus "
        "belum panas. Kabelnya agak pendek jadi perlu colokan dekat meja.\"*"
    )
    st.markdown(
        "  - Prediksi model: **Real**, Prob Real ≈ **99.93%**, Prob Fake ≈ **0.07%**. "
        "Model sangat yakin review ini Real karena berisi kelebihan dan kekurangan yang seimbang, "
        "detail pengalaman, dan tidak ada bahasa promosi berlebihan."
    )
    st.markdown(
        "- **Contoh Fake**: *\"Best seller banget, kualitas premium, dijamin puas pokoknya!\"*"
    )
    st.markdown(
        "  - Prediksi model: **Fake**, Prob Real ≈ **1.81%**, Prob Fake ≈ **98.19%**. "
        "Model sangat yakin review ini Fake karena pola kalimat mirip iklan, banyak hiperbola, "
        "dan tidak ada detail spesifik mengenai pengalaman penggunaan."
    )

# ---------- Tab Tentang ----------
with tab_about:
    st.markdown("### Tentang aplikasi")
    st.write(
        "Aplikasi ini merupakan proyek skripsi mahasiswa Program Studi Data Science "
        "yang berfokus pada pendeteksian fake review di e-commerce Indonesia."
    )
    st.write(
        "Model utama yang digunakan adalah IndoBERT yang telah di-fine-tune untuk klasifikasi "
        "review Real/Fake. Data dikumpulkan dengan web scraping ulasan produk dari Tokopedia dan "
        "Shopee, kemudian melalui proses pembersihan teks, pelabelan (gold dan pseudo label), "
        "serta pembagian data latih dan uji."
    )
    st.write(
        "Antarmuka ini dibangun dengan Streamlit dan memuat model IndoBERT yang tersimpan di "
        "Hugging Face Hub, sehingga pengguna dapat memasukkan review dalam Bahasa Indonesia dan "
        "melihat probabilitas Real/Fake, tingkat kepercayaan model, serta reasoning sederhana "
        "yang menjelaskan pola bahasa pada review tersebut."
    )
    st.write(
        "Aplikasi ini ditujukan sebagai bukti konsep bahwa pendekatan deep learning berbasis "
        "Transformer dapat membantu mengidentifikasi potensi fake review dan mendukung "
        "pengambilan keputusan yang lebih informatif bagi pengguna e-commerce."
    )

