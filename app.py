import streamlit as st
import joblib
import os

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Analisis Sentimen BBM Etanol",
    page_icon="⛽",
    layout="centered"
)

# ==========================
# Debug (boleh dihapus setelah deploy berhasil)
# ==========================
st.write("Python berjalan")
st.write("Isi folder:", os.listdir())

# ==========================
# Load Model
# ==========================
try:
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    st.success("✅ Model dan TF-IDF berhasil dimuat")

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==========================
# CSS
# ==========================
st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

.title-box{
    background: linear-gradient(90deg,#0d6efd,#198754);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
}

.info-box{
    background:#ffffff;
    padding:15px;
    border-radius:10px;
    border-left:6px solid #0d6efd;
    margin-top:10px;
    margin-bottom:20px;
}

.result-box{
    padding:18px;
    border-radius:10px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.markdown("""
<div class="title-box">
<h1>⛽ Analisis Sentimen BBM Etanol</h1>
<h4>Metode Logistic Regression</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Aplikasi ini digunakan untuk melakukan klasifikasi sentimen komentar masyarakat terhadap program BBM Etanol menjadi <b>Positif</b>, <b>Netral</b>, atau <b>Negatif</b>.
</div>
""", unsafe_allow_html=True)

# ==========================
# Input
# ==========================
komentar = st.text_area(
    "Masukkan Komentar",
    placeholder="Contoh: Program BBM Etanol sangat bagus dan ramah lingkungan..."
)

# ==========================
# Prediksi
# ==========================
if st.button("🔍 Analisis Sentimen", use_container_width=True):

    if komentar.strip() == "":
        st.warning("Silakan masukkan komentar terlebih dahulu.")

    else:

        # TF-IDF
        data = tfidf.transform([komentar])

        # Prediksi
        prediksi = model.predict(data)[0]
        probabilitas = model.predict_proba(data)[0]

        # Hasil
        if prediksi == -1:

            st.markdown("""
            <div class="result-box" style="background:#f8d7da;color:#b02a37;">
            🔴 NEGATIF
            </div>
            """, unsafe_allow_html=True)

        elif prediksi == 0:

            st.markdown("""
            <div class="result-box" style="background:#fff3cd;color:#856404;">
            🟡 NETRAL
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-box" style="background:#d1e7dd;color:#146c43;">
            🟢 POSITIF
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.subheader("📊 Probabilitas Prediksi")

        st.write("Negatif")
        st.progress(float(probabilitas[0]))
        st.write(f"**{probabilitas[0]*100:.2f}%**")

        st.write("Netral")
        st.progress(float(probabilitas[1]))
        st.write(f"**{probabilitas[1]*100:.2f}%**")

        st.write("Positif")
        st.progress(float(probabilitas[2]))
        st.write(f"**{probabilitas[2]*100:.2f}%**")

# ==========================
# Footer
# ==========================
st.markdown("""
<div class="footer">
Dibuat untuk penelitian Analisis Sentimen BBM Etanol menggunakan Logistic Regression
</div>
""", unsafe_allow_html=True)
