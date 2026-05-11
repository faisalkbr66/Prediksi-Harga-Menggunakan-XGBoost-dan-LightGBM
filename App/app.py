import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Kalkulator Harga Mobil Bekas",
    page_icon="8308414.png",
    layout="centered"
)

# ==========================================
# 2. MEMUAT MODEL DAN ENCODER
# ==========================================
# Menggunakan @st.cache_resource agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_assets():
    xgb_model = joblib.load('xgboost_model.pkl')
    lgbm_model = joblib.load('lightgbm_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return xgb_model, lgbm_model, encoders

try:
    xgb_model, lgbm_model, encoders = load_assets()
except FileNotFoundError:
    st.error("⚠️ File model (.pkl) tidak ditemukan. Pastikan 'xgboost_model.pkl', 'lightgbm_model.pkl', dan 'label_encoders.pkl' ada di folder yang sama dengan app.py.")
    st.stop()

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("AI Prediksi Harga Mobil Bekas")
st.markdown("""
Aplikasi ini menggunakan Kecerdasan Buatan (AI) untuk memprediksi harga jual mobil bekas berdasarkan spesifikasinya. 
Silakan masukkan detail kendaraan di bawah ini.
""")

st.divider()

# Membagi layar menjadi 2 kolom untuk tampilan yang lebih rapi
col1, col2 = st.columns(2)

with col1:
    st.subheader("Informasi Dasar")
    # Mengambil daftar pilihan dari Label Encoder agar sesuai dengan data training
    brand = st.selectbox("Merek Mobil", encoders['brand'].classes_)
    tahun = st.number_input("Tahun Pembuatan", min_value=1990, max_value=datetime.now().year, value=2018, step=1)
    km_driven = st.number_input("Jarak Tempuh (Kilometer)", min_value=0, value=50000, step=1000)
    owner = st.selectbox("Kepemilikan Tangan Ke-", encoders['owner'].classes_)
    seller_type = st.selectbox("Dijual Oleh", encoders['seller_type'].classes_)

with col2:
    st.subheader("Spesifikasi Mesin")
    fuel = st.selectbox("Jenis Bahan Bakar", encoders['fuel'].classes_)
    transmission = st.selectbox("Transmisi", encoders['transmission'].classes_)
    mileage = st.number_input("Konsumsi BBM (km/liter)", min_value=0.0, value=18.0, step=0.1)
    engine = st.number_input("Kapasitas Mesin (CC)", min_value=500, max_value=8000, value=1200, step=100)
    max_power = st.number_input("Tenaga Maksimal (BHP)", min_value=10.0, max_value=1000.0, value=80.0, step=1.0)
    seats = st.number_input("Jumlah Kursi", min_value=2, max_value=14, value=5, step=1)

st.divider()

# Pilihan Model AI
st.subheader("Pengaturan AI")
model_choice = st.radio("Pilih Mesin Prediksi (Model AI):", ["LightGBM (Disarankan)", "XGBoost"], horizontal=True)

# Tombol Prediksi
if st.button("Hitung Prediksi Harga", use_container_width=True, type="primary"):
    
    # ==========================================
    # 4. PEMROSESAN DATA (PREPROCESSING)
    # ==========================================
    # Menghitung umur mobil sesuai logika di notebook: datetime.now().year - year
    current_year = datetime.now().year
    age = current_year - tahun

    # Menyusun data ke dalam DataFrame
    input_data = pd.DataFrame({
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'brand': [brand],
        'age': [age]
    })

    # Mengubah data teks menjadi angka (Label Encoding) seperti saat training
    categorical_cols = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']
    for col in categorical_cols:
        # Menggunakan transform dari encoder yang sudah dilatih
        input_data[col] = encoders[col].transform(input_data[col])
        # Mengubah ke tipe 'category' karena LightGBM sangat bergantung pada format ini
        input_data[col] = input_data[col].astype('category')

    # Memastikan urutan kolom sama persis dengan X_train di notebook
    expected_columns = ['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'brand', 'age']
    input_data = input_data[expected_columns]

    # ==========================================
    # 5. PREDIKSI
    # ==========================================
    with st.spinner('AI sedang menganalisis harga pasar...'):
        if "LightGBM" in model_choice:
            pred_log = lgbm_model.predict(input_data)
        else:
            # XGBoost di notebook Anda mungkin membutuhkan format khusus jika kategori diaktifkan
            pred_log = xgb_model.predict(input_data)
        
        # Karena di notebook target di-log menggunakan np.log1p, kita kembalikan menggunakan np.expm1
        pred_price = np.expm1(pred_log)[0]

    # ==========================================
    # 6. MENAMPILKAN HASIL
    # ==========================================
    st.success("Prediksi Selesai!")
    
    # Menampilkan kartu hasil prediksi dengan format mata uang
    st.markdown(f"""
    <div style="padding:20px; border-radius:10px; text-align:center;">
        <h3 style="margin-bottom:0;">Estimasi Harga Jual:</h3>
        <h1 style="color:#2e7bcf; margin-top:5px;">USD {pred_price:,.0f}</h1>
        <p style="font-size:14px; "><i>*Harga dapat bervariasi tergantung kondisi fisik dan pajak kendaraan.</i></p>
    </div>
    """, unsafe_allow_html=True)