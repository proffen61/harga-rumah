import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Memuat dataset
df = pd.read_csv('AmesHousing.csv')  # Pastikan path sesuai dengan dataset Anda

# Pisahkan fitur dan target
X = df[['Overall Qual', 'Gr Liv Area', 'TotRms AbvGrd', 'Year Built']]  # Fitur yang dipilih
y = df['SalePrice']  # Target: 'SalePrice'

# Mengisi missing values dengan rata-rata untuk fitur numerik
X = X.fillna(X.mean())  # Mengisi missing values untuk fitur numerik

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Linear Regression
lr_model = LinearRegression()

# Preprocessing untuk fitur numerik: imputasi dan standarisasi
numeric_features = ['Overall Qual', 'Gr Liv Area', 'TotRms AbvGrd', 'Year Built']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputasi missing values dengan rata-rata
    ('scaler', StandardScaler())  # Standarisasi fitur numerik
])

# Preprocessing untuk fitur kategorikal (jika ada fitur kategorikal di dataset)
# Untuk dataset ini, kita tidak menggunakan fitur kategorikal, jadi ini hanya untuk contoh
categorical_features = []  # Tidak ada fitur kategorikal dalam subset ini
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Gabungkan transformasi numerik dan kategorikal dengan ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Gabungkan preprocessor dengan model dalam pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', lr_model)
])

# Latih model
pipeline.fit(X_train, y_train)

# Simpan model dan preprocessor
joblib.dump(pipeline, 'linear_regression_model.pkl')  # Menyimpan pipeline (model + preprocessor)
