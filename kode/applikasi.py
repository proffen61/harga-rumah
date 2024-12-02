import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Menyertakan pipeline yang sudah dilatih sebelumnya
try:
    # Memuat pipeline yang sudah disimpan (model + preprocessor)
    pipeline = joblib.load('linear_regression_model.pkl')  # Pipeline yang sudah dilatih
except FileNotFoundError as e:
    st.error(f"File pipeline tidak ditemukan: {e}")
    st.stop()  # Menghentikan eksekusi lebih lanjut jika file tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat pipeline: {e}")
    st.stop()

# Judul aplikasi
st.title('Prediksi Harga Rumah dengan Machine Learning')

# Sidebar untuk navigasi
st.sidebar.title('Navigasi')
options = st.sidebar.radio("Pilih Halaman", ['Prediksi Harga Rumah', 'Analisis Data', 'Evaluasi Model', 'Insight Model'])

# Memuat dataset
df = pd.read_csv('AmesHousing.csv')  # Ganti dengan path dataset yang sesuai

# Fungsi untuk menampilkan tabel data
def show_data():
    st.header("Data Rumah")
    st.write("Menampilkan beberapa baris pertama dari dataset:")
    st.write(df.head())

# Fungsi untuk visualisasi distribusi harga rumah
def show_price_distribution():
    st.header("Distribusi Harga Rumah")
    plt.figure(figsize=(8, 6))
    sns.histplot(df['SalePrice'], kde=True, color='blue')
    plt.title('Distribusi Harga Rumah')
    plt.xlabel('Harga Rumah')
    plt.ylabel('Frekuensi')
    st.pyplot(plt)
    plt.clf()

# Fungsi untuk visualisasi korelasi
def show_correlation():
    st.header("Korelasi antara Fitur dan Harga Rumah")
    # Pilih hanya kolom numerik untuk analisis korelasi
    df_numeric = df.select_dtypes(include=['number'])  # Mengambil hanya kolom numerik

    # Mengisi NaN hanya pada kolom numerik menggunakan mean kolom tersebut
    df_numeric_filled = df_numeric.fillna(df_numeric.mean())  # Mengisi NaN dengan rata-rata

    # Menghitung korelasi
    corr_matrix = df_numeric_filled.corr()

    # Visualisasi heatmap korelasi
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix[['SalePrice']], annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Korelasi Fitur dengan Harga Rumah')
    st.pyplot(plt)
    plt.clf()

# Fungsi untuk evaluasi model
def show_model_evaluation():
    st.header("Evaluasi Model")
    X = df[['Overall Qual', 'Gr Liv Area', 'TotRms AbvGrd', 'Year Built']]
    y = df['SalePrice']

    X = X.select_dtypes(include=['float64', 'int64']).fillna(X.mean())
    y = y.fillna(y.mean())  # Mengisi nilai yang hilang pada target dengan rata-rata

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train_transformed, y_train)

    y_pred = lr_model.predict(X_test_transformed)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'Root Mean Squared Error: {rmse:.2f}')
    st.write(f'R^2 Score: {r2:.2f}')

# Fungsi untuk prediksi harga rumah
def show_predict():
    st.header('Masukkan Fitur Rumah')
    
    # Input fitur rumah dari user
    overall_qual = st.slider('Overall Quality (1-10)', 1, 10)
    gr_liv_area = st.number_input('Gr Liv Area (Ukuran Rumah dalam SqFt)', min_value=0, step=1)
    tot_rms = st.number_input('Total Rooms (Total Ruangan dalam Rumah)', min_value=0, step=1)
    year_built = st.number_input('Tahun Dibangun', min_value=1800, max_value=2024, step=1)

    # Membuat input fitur dalam format DataFrame, dengan nama kolom yang sesuai dengan model
    input_features = pd.DataFrame([[overall_qual, gr_liv_area, tot_rms, year_built]],
                                  columns=['Overall Qual', 'Gr Liv Area', 'TotRms AbvGrd', 'Year Built'])
    
    # Menampilkan data input untuk verifikasi
    st.write("Fitur Rumah yang Dimasukkan:")
    st.write(input_features)

    if st.button('Prediksi Harga Rumah'):
        try:
            # Pastikan input_features adalah DataFrame dengan nama kolom yang benar
            # Prediksi harga rumah menggunakan pipeline
            prediction = pipeline.predict(input_features)
            
            st.write(f'Prediksi Harga Rumah: Rp {prediction[0]:,.2f}')
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# Fungsi untuk insight model
def show_model_insight():
    st.header('Insight Model Prediksi')
    st.write("""  
    Model Linear Regression ini menggunakan beberapa fitur rumah untuk memprediksi harga jual rumah. 
    Fitur yang digunakan antara lain:
    - **Overall Quality**: Kualitas keseluruhan rumah.
    - **Gr Liv Area**: Ukuran rumah dalam square feet.
    - **TotRms AbvGrd**: Total ruangan di atas tanah.
    - **Year Built**: Tahun rumah dibangun.
    
    Model ini mengasumsikan bahwa hubungan antara fitur-fitur ini dan harga rumah adalah linier, 
    yang dapat memberikan prediksi harga berdasarkan input yang diberikan.
    """)
    st.write("""  
    **Evaluasi Model:**
    - Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE) digunakan untuk mengukur akurasi prediksi.
    - R-squared (R²) menunjukkan seberapa baik model menjelaskan variabilitas data.
    """)

# Menampilkan halaman berdasarkan pilihan sidebar
if options == 'Prediksi Harga Rumah':
    show_predict()
elif options == 'Analisis Data':
    show_data()
    show_price_distribution()
    show_correlation()  # Menampilkan korelasi setelah data dan distribusi
elif options == 'Evaluasi Model':
    show_model_evaluation()
elif options == 'Insight Model':
    show_model_insight()
