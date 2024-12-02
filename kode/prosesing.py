import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Memuat dataset
df = pd.read_csv('AmesHousing.csv')  # Ganti dengan path dataset yang sesuai

# Memisahkan fitur (X) dan target (y)
X = df[['Overall Qual', 'Gr Liv Area', 'TotRms AbvGrd', 'Year Built']]  # Kolom relevan yang dipilih
y = df['SalePrice']  # Target: 'SalePrice'

# Identifikasi kolom numerik dan kategorikal
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Membuat pipeline untuk preprocessing data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Mengimputasi nilai yang hilang dengan median
    ('scaler', StandardScaler())])  # Standarisasi numerik

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Mengimputasi nilai yang hilang dengan modus
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # One hot encoding untuk fitur kategorikal

# Gabungkan transformasi numerik dan kategorikal dengan ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Fit transformer pada data training
X_transformed = preprocessor.fit_transform(X)

# Simpan preprocessor ke file menggunakan joblib
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Preprocessor berhasil disimpan ke 'preprocessor.pkl'")
