import streamlit as st
import joblib
import pandas as pd

# Memuat pipeline (model + preprocessor) yang sudah disimpan
try:
    pipeline = joblib.load('linear_regression_model.pkl')
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# Judul aplikasi
st.title('Prediksi Harga Rumah dengan Machine Learning')

# Input fitur rumah dari user
overall_qual = st.slider('Overall Quality (1-10)', 1, 10)
gr_liv_area = st.number_input('Gr Liv Area (Ukuran Rumah dalam SqFt)', min_value=0, step=1)
tot_rms = st.number_input('Total Rooms (Total Ruangan dalam Rumah)', min_value=0, step=1)
year_built = st.number_input('Tahun Dibangun', min_value=1800, max_value=2024, step=1)

# Membuat input fitur dalam format DataFrame
input_features = pd.DataFrame([[overall_qual, gr_liv_area, tot_rms, year_built]],
                              columns=['Overall Qual', 'Gr Liv Area', 'TotRms AbvGrd', 'Year Built'])

# Menampilkan data input untuk verifikasi
st.write("Fitur Rumah yang Dimasukkan:")
st.write(input_features)

if st.button('Prediksi Harga Rumah'):
    try:
        # Prediksi harga rumah menggunakan pipeline (model + preprocessor)
        prediction = pipeline.predict(input_features)
        st.write(f'Prediksi Harga Rumah: Rp {prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
