import streamlit as st
import pandas as pd
import joblib
import warnings

# Mengabaikan peringatan versi Scikit-learn yang mungkin tidak cocok
warnings.filterwarnings("ignore", category=UserWarning)

# --- KONFIGURASI HALAMAN ---
# Mengatur judul tab, ikon, dan tata letak. Ini harus menjadi perintah Streamlit pertama.
st.set_page_config(
    page_title="Prediksi Rating Sepatu",
    page_icon="👟",
    layout="centered" 
)

# --- FUNGSI UNTUK GAYA TAMPILAN (CSS) ---
def local_css():
    """Menyisipkan CSS kustom untuk meniru tampilan dari file HTML."""
    st.markdown("""
        <style>
        /* Import Google Font 'Inter' */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Mengubah font utama Streamlit */
        html, body, [class*="st-"], [class*="css-"] {
            font-family: 'Inter', sans-serif;
        }

        /* Memberi background gradien seperti di HTML */
        .stApp {
            background-image: linear-gradient(to bottom right, #f8fafc, #e0e7ff);
            background-attachment: fixed;
            background-size: cover;
        }

        /* Styling untuk judul utama */
        h1 {
            font-weight: 700 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- FUNGSI PEMUATAN DENGAN CACHING (TETAP SAMA, KARENA EFISIEN) ---

@st.cache_resource
def load_model():
    """Memuat pipeline model yang sudah termasuk preprocessor."""
    try:
        model = joblib.load("shoe_rating_predictor_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.error("File model 'shoe_rating_predictor_pipeline.pkl' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

@st.cache_data
def load_unique_brands():
    """Memuat daftar merek dari file yang sudah diproses (lebih efisien)."""
    try:
        with open("unique_brands.txt", "r") as f:
            brands = [line.strip() for line in f]
        return sorted(brands)
    except FileNotFoundError:
        return sorted(['ASIAN', 'Reebok', 'Puma', 'Adidas', 'Bata'])

# --- MEMUAT DATA DAN MODEL ---
model_pipeline = load_model()
brands = load_unique_brands()
student_name = "Fajriya Hakim" # Ganti dengan nama Anda

# --- APLIKASI UTAMA ---

# Terapkan CSS kustom
local_css()

# Header aplikasi
st.title("Prediksi Rating Sepatu")
st.markdown(f"Proyek UAS oleh: **{student_name}**")
st.write("---") # Garis pemisah

# Form prediksi yang meniru desain HTML
with st.form("prediction_form"):
    st.subheader("Masukkan Detail Sepatu")
    
    # Input field yang sesuai dengan HTML
    brand = st.selectbox("Merek Sepatu", brands)
    how_many_sold = st.number_input("Jumlah Terjual", min_value=0, step=1, placeholder="Contoh: 500")
    current_price = st.number_input("Harga Saat Ini (dalam Rupee)", min_value=0.0, step=50.0, format="%.2f", placeholder="Contoh: 1099.00")

    # Tombol submit
    submitted = st.form_submit_button("✨ Prediksi Rating")

# Logika setelah form disubmit
if submitted:
    if model_pipeline:
        input_data = pd.DataFrame({
            'Brand_Name': [brand],
            'How_Many_Sold': [how_many_sold],
            'Current_Price': [current_price]
        })

        try:
            # Prediksi menggunakan pipeline
            prediction = model_pipeline.predict(input_data)
            rating = round(prediction[0], 2)
            
            # Tampilkan hasil dengan format sukses (background hijau)
            st.success(f"**Hasil Prediksi Rating: {rating} / 5.0**")
            
            # Menambahkan progress bar sebagai visualisasi rating
            st.progress(rating / 5.0)

        except Exception as e:
            # Tampilkan hasil dengan format error (background merah)
            st.error(f"Terjadi Error: {e}")
            st.info("Pastikan model yang dimuat adalah pipeline lengkap dan input sudah benar.")

    else:
        # Jika model tidak berhasil dimuat
        st.error("Model tidak dapat dimuat. Prediksi gagal.")