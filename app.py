import streamlit as st
import joblib

# =====================================================
# Konfigurasi Halaman
# =====================================================
st.set_page_config(
    page_title="Analisis Sentimen BBM Etanol",
    page_icon="⛽",
    layout="centered"
)

# =====================================================
# Load Model
# =====================================================
try:
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>

.main{
    background:#f8f9fa;
}

.title-box{
    background: linear-gradient(90deg,#0d6efd,#198754);
    padding:22px;
    border-radius:15px;
    color:white;
    text-align:center;
}

.info-box{
    background:white;
    padding:18px;
    border-radius:10px;
    border-left:6px solid #0d6efd;
    margin-top:15px;
    margin-bottom:15px;
}

.result-box{
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:26px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# Header
# =====================================================
st.markdown("""
<div class="title-box">
<h1>⛽ Analisis Sentimen BBM Etanol</h1>
<h3>Menggunakan Metode Logistic Regression</h3>
</div>
""", unsafe_allow_html=True)

# =====================================================
# Informasi Penelitian
# =====================================================

st.markdown("""
<div class="info-box">

### Informasi Penelitian

- **Metode Klasifikasi :** Logistic Regression
- **Ekstraksi Fitur :** TF-IDF
- **Penanganan Data Tidak Seimbang :** SMOTE
- **Skenario Terbaik :** Split Data 90% Training : 10% Testing

Masukkan komentar mengenai program BBM Etanol, kemudian sistem akan mengklasifikasikan sentimen menjadi Positif, Netral, atau Negatif.

</div>
""", unsafe_allow_html=True)

# =====================================================
# Input
# =====================================================

komentar = st.text_area(
    "Masukkan Komentar",
    placeholder="Contoh : Program BBM Etanol sangat bagus dan ramah lingkungan."
)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Analisis Sentimen", use_container_width=True):

        if komentar.strip() == "":
            st.warning("Silakan masukkan komentar terlebih dahulu.")

        else:

            data = tfidf.transform([komentar])

            prediksi = model.predict(data)[0]
            probabilitas = model.predict_proba(data)[0]

            if prediksi == -1:

                st.markdown("""
                <div class="result-box" style="background:#f8d7da;color:#842029;">
                🔴 NEGATIF
                </div>
                """, unsafe_allow_html=True)

            elif prediksi == 0:

                st.markdown("""
                <div class="result-box" style="background:#fff3cd;color:#664d03;">
                🟡 NETRAL
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-box" style="background:#d1e7dd;color:#0f5132;">
                🟢 POSITIF
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            st.subheader("📊 Probabilitas Prediksi")

            st.write("Negatif")
            st.progress(float(probabilitas[0]))
            st.write(f"{probabilitas[0]*100:.2f}%")

            st.write("Netral")
            st.progress(float(probabilitas[1]))
            st.write(f"{probabilitas[1]*100:.2f}%")

            st.write("Positif")
            st.progress(float(probabilitas[2]))
            st.write(f"{probabilitas[2]*100:.2f}%")

with col2:
    if st.button("🗑 Bersihkan", use_container_width=True):
        st.rerun()



st.divider()



# =====================================================
# Footer
# =====================================================

st.markdown("""

""", unsafe_allow_html=True)
