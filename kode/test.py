import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Memuat dataset (gunakan path yang sesuai jika tidak di direktori yang sama)
df = pd.read_csv('AmesHousing.csv')  # Ganti dengan path dataset Anda

# Pisahkan fitur dan target
X = df[['Overall Qual', 'Gr Liv Area', 'TotRms AbvGrd', 'Year Built']]  # Kolom relevan yang dipilih
y = df['SalePrice']  # Target: 'SalePrice'

# Pastikan hanya kolom numerik
X = X.select_dtypes(include=['float64', 'int64'])

# Tangani missing values pada fitur
X = X.fillna(X.mean())  # Mengisi nilai yang hilang dengan rata-rata kolom

# Tangani missing values pada target (jika ada)
y = y.fillna(y.mean())  # Mengisi nilai yang hilang pada target dengan rata-rata

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi scaler untuk standariasi fitur numerik
scaler = StandardScaler()

# Membuat dan melatih model Linear Regression
lr_model = LinearRegression()

# Buat pipeline untuk preprocessing dan model
pipeline = Pipeline(steps=[
    ('scaler', scaler),  # Standarisasi fitur numerik
    ('model', lr_model)   # Linear Regression Model
])

# Latih model
pipeline.fit(X_train, y_train)

# Simpan model dan preprocessor menggunakan joblib
joblib.dump(pipeline, 'linear_regression_model.pkl')  # Menyimpan model dan preprocessor dalam satu file
