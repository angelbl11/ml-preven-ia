import pandas as pd
import numpy as np
import os
import glob
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglehub

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Create datasets directory if it doesn't exist
if not os.path.exists('datasets'):
    os.makedirs('datasets')

# Download datasets
print("Downloading datasets...")
try:
    print("Downloading Dataset 1...")
    api.dataset_download_files('simaanjali/diabetes-classification-dataset',
                               path='datasets/diabetes_class',
                               unzip=True)
    print("Downloading Dataset 2...")
    api.dataset_download_files('iammustafatz/diabetes-prediction-dataset',
                               path='datasets/diabetes_pred',
                               unzip=True)
    print("Downloading Dataset 3...")
    api.dataset_download_files('khan1803115/hypertension-risk-model-main',
                               path='datasets/hypertension',
                               unzip=True)
    print("Downloading Dataset 4 (Obesity)...")
    path = kagglehub.dataset_download(
        "adeniranstephen/obesity-prediction-dataset")
    print("Path to obesity dataset files:", path)

except Exception as e:
    print(f"Error downloading datasets: {str(e)}")


def find_csv_file(directory):
    """Find any CSV file in the given directory"""
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if csv_files:
        return csv_files[0]  # Return the first CSV found
    return None


def load_dataset(directory, name):
    csv_path = find_csv_file(directory)
    if csv_path is None:
        print(f"Warning: Could not find any CSV file in {name}")
        return None
    print(f"Loading {name} from: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {name}: {str(e)}")
        return None


# Load datasets
df1 = load_dataset('datasets/diabetes_class', "Dataset 1")
df2 = load_dataset('datasets/diabetes_pred', "Dataset 2")
df3 = load_dataset('datasets/hypertension', "Dataset 3")
df4 = load_dataset(path, "Dataset 4 (Obesity)")

# Print dataset information
for i, df in enumerate([df1, df2, df3, df4], 1):
    if df is not None:
        print(f"\nDataset {i} shape:", df.shape)
        print(f"Dataset {i} columns:", df.columns.tolist())

# Standardize column names (mapping to your required variables)
column_mapping = {
    # Dataset 1 specific
    'bmi': 'bmi',
    'ldl': 'ldl',
    'tg': 'triglycerides',
    'gender': 'gender',
    'age': 'age',

    # Dataset 2 specific
    'blood_glucose_level': 'fasting_glucose',
    'hba1c_level': 'hba1c',

    # Dataset 3 specific
    'male': 'gender',
    'sysbp': 'systolic_bp',
    'diabp': 'diastolic_bp',
    'glucose': 'fasting_glucose',
    'currentsmoker': 'smoker',
    'cigsperday': 'cig_per_day',

    # Dataset 4 (Obesity) specific
    'physical_activity_frequency': 'physical_activity',
    'consumption_of_alcohol': 'alcohol_consumption'
}


def classify_smoker_intensity(df):
    """
    Classify smokers based on cigarettes per day from dataset
    """
    if 'smoker' in df.columns and 'cig_per_day' in df.columns:
        # Clasificar intensidad basado en los datos existentes
        conditions = [
            (df['smoker'] == 1) & (df['cig_per_day'] <= 10),
            (df['smoker'] == 1) & (df['cig_per_day'] <= 20),
            (df['smoker'] == 1) & (df['cig_per_day'] > 20),
            (df['smoker'] == 0)
        ]
        choices = ['light', 'moderate', 'heavy', 'non_smoker']
        df['smoking_intensity'] = np.select(
            conditions, choices, default='non_smoker')
    elif 'smoker' in df.columns:
        # Si solo tenemos el estado de fumador pero no la cantidad
        df['smoking_intensity'] = np.where(
            df['smoker'] == 1, 'unknown', 'non_smoker')

    return df


def standardize_columns(df):
    if df is None:
        return None

    print("\nOriginal columns:", df.columns.tolist())

    # Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    print("Lowercase columns:", df.columns.tolist())

    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()

    # Rename columns based on mapping
    df_renamed = df.rename(columns=column_mapping)

    # Handle gender column
    if 'male' in df.columns:
        df_renamed['gender'] = df['male'].map({1: 'male', 0: 'female'})
    elif 'gender' in df.columns:
        df_renamed['gender'] = df_renamed['gender'].str.lower()
        df_renamed['gender'] = df_renamed['gender'].map(
            {'male': 'male', 'female': 'female', 'm': 'male', 'f': 'female'})

    # Handle smoking status if present
    if 'smoker' in df_renamed.columns:
        smoking_status_map = {
            'yes': 1, 'no': 0,
            'current': 1, 'never': 0, 'former': 1,
            'smoker': 1, 'non-smoker': 0,
            1: 1, 0: 0,
            'true': 1, 'false': 0
        }
        if isinstance(df_renamed['smoker'].iloc[0], str):
            df_renamed['smoker'] = df_renamed['smoker'].str.lower()
        df_renamed['smoker'] = df_renamed['smoker'].map(smoking_status_map)

    # Lista de columnas requeridas
    required_cols = [
        'bmi', 'ldl', 'triglycerides', 'gender', 'age',
        'fasting_glucose', 'hba1c', 'systolic_bp', 'diastolic_bp',
        'smoker', 'cig_per_day', 'physical_activity', 'alcohol_consumption',
        'family_history_diabetes', 'family_history_obesity', 'family_history_hypertension'
    ]

    # Seleccionar solo las columnas que existen y son requeridas
    existing_cols = [col for col in required_cols if col in df_renamed.columns]
    df_final = df_renamed[existing_cols].copy()

    # Convert numeric columns
    numeric_cols = ['bmi', 'ldl', 'triglycerides', 'fasting_glucose', 'hba1c',
                    'systolic_bp', 'diastolic_bp', 'age', 'cig_per_day']
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    print("Final columns:", df_final.columns.tolist())
    return df_final


def generate_mexican_health_data(df):
    """
    Generate health conditions and family history based on Mexican epidemiological data
    """
    n_samples = len(df)

    # Generar o ajustar género si no existe
    if 'gender' not in df.columns:
        df['gender'] = np.random.choice(
            ['male', 'female'], size=n_samples, p=[0.48, 0.52])

    # Generar edad entre 20-79 años (rango más común en los estudios)
    if 'age' not in df.columns:
        df['age'] = np.random.normal(45, 15, n_samples)  # media 45, std 15
        df['age'] = df['age'].clip(20, 79)

    # Prevalencias base según datos mexicanos más recientes
    base_prob_diabetes = 0.183  # 18.3% (2022)
    base_prob_obesity = 0.369   # 36.9% (2022)
    base_prob_hypertension = 0.494  # 49.4% (AHA 2020)

    # Generar historiales familiares considerando heredabilidad y coocurrencias
    # Basado en las tablas epidemiológicas proporcionadas

    # Historia familiar de diabetes (considerando la alta prevalencia en México)
    df['family_history_diabetes'] = np.random.binomial(
        1, base_prob_diabetes * 1.5, n_samples)

    # Historia familiar de obesidad (considerando la alta prevalencia)
    df['family_history_obesity'] = np.random.binomial(
        1, base_prob_obesity * 1.3, n_samples)

    # Historia familiar de hipertensión
    df['family_history_hypertension'] = np.random.binomial(
        1, base_prob_hypertension * 1.2, n_samples)

    # Ajustar coocurrencias basadas en la tabla epidemiológica
    # Si hay historia familiar de obesidad, ajustar probabilidades de diabetes e hipertensión
    obesity_mask = df['family_history_obesity'] == 1
    df.loc[obesity_mask, 'family_history_diabetes'] = np.random.binomial(
        1, 0.17 * 1.7, obesity_mask.sum())  # 1.7 veces más probable
    df.loc[obesity_mask, 'family_history_hypertension'] = np.random.binomial(
        1, 0.588, obesity_mask.sum())  # 58.8% coocurrencia

    # Si hay historia familiar de diabetes, ajustar probabilidad de hipertensión
    diabetes_mask = df['family_history_diabetes'] == 1
    df.loc[diabetes_mask, 'family_history_hypertension'] = np.random.binomial(
        1, 0.462, diabetes_mask.sum())  # 46.2% coocurrencia

    return df


def add_lifestyle_factors(df):
    """
    Add physical activity and alcohol consumption variables with Mexican population patterns
    """
    n_samples = len(df)

    # Physical activity frequency
    if 'physical_activity' not in df.columns:
        # Basado en datos de sedentarismo en México
        df['physical_activity'] = np.random.choice(
            ['none', 'light', 'moderate', 'frequent'],
            size=n_samples,
            p=[0.40, 0.30, 0.20, 0.10]  # 40% sedentarios según ENSANUT
        )

    # Alcohol consumption
    if 'alcohol_consumption' not in df.columns:
        df['alcohol_consumption'] = np.random.choice(
            ['none', 'light', 'moderate', 'heavy'],
            size=n_samples,
            p=[0.45, 0.30, 0.15, 0.10]
        )

    return df


# Actualizar el proceso de combinación de datasets
print("\nStandardizing Dataset 1:")
df1_std = standardize_columns(df1)
print("\nStandardizing Dataset 2:")
df2_std = standardize_columns(df2)
print("\nStandardizing Dataset 3:")
df3_std = standardize_columns(df3)
print("\nStandardizing Dataset 4:")
df4_std = standardize_columns(df4)

# Combine datasets
dfs_to_combine = []
if df1_std is not None and len(df1_std) > 0:
    df1_std = generate_mexican_health_data(df1_std)
    df1_std = classify_smoker_intensity(df1_std)
    df1_std = add_lifestyle_factors(df1_std)
    dfs_to_combine.append(df1_std)
if df2_std is not None and len(df2_std) > 0:
    df2_std = generate_mexican_health_data(df2_std)
    df2_std = classify_smoker_intensity(df2_std)
    df2_std = add_lifestyle_factors(df2_std)
    dfs_to_combine.append(df2_std)
if df3_std is not None and len(df3_std) > 0:
    df3_std = generate_mexican_health_data(df3_std)
    df3_std = classify_smoker_intensity(df3_std)
    df3_std = add_lifestyle_factors(df3_std)
    dfs_to_combine.append(df3_std)
if df4_std is not None and len(df4_std) > 0:
    df4_std = generate_mexican_health_data(df4_std)
    df4_std = classify_smoker_intensity(df4_std)
    df4_std = add_lifestyle_factors(df4_std)
    dfs_to_combine.append(df4_std)

# Después de combinar los datasets, asegurar que solo tenemos las columnas necesarias
if not dfs_to_combine:
    raise ValueError("No valid datasets to combine!")

# Combine all datasets
combined_df = pd.concat(dfs_to_combine, ignore_index=True)

# Lista final de columnas requeridas
final_required_cols = [
    'bmi', 'ldl', 'triglycerides', 'gender', 'age',
    'fasting_glucose', 'hba1c', 'systolic_bp', 'diastolic_bp',
    'smoker', 'cig_per_day', 'physical_activity', 'alcohol_consumption',
    'family_history_diabetes', 'family_history_obesity', 'family_history_hypertension'
]

# Mantener solo las columnas requeridas en el dataset final
final_df = combined_df[final_required_cols].copy()

# Fill missing values with the mean for numeric columns and mode for categorical
numeric_cols = ['bmi', 'ldl', 'triglycerides', 'fasting_glucose', 'hba1c',
                'systolic_bp', 'diastolic_bp', 'age', 'cig_per_day']
for col in numeric_cols:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        final_df[col].fillna(final_df[col].mean(), inplace=True)

# Fill categorical variables
categorical_cols = ['gender', 'physical_activity', 'alcohol_consumption',
                    'family_history_diabetes', 'family_history_obesity', 'family_history_hypertension']
for col in categorical_cols:
    if col in final_df.columns:
        final_df[col].fillna(final_df[col].mode()[0], inplace=True)

# Ensure smoker is properly filled
if 'smoker' in final_df.columns:
    final_df['smoker'] = final_df['smoker'].fillna(0)
    final_df['cig_per_day'] = final_df['cig_per_day'].fillna(0)

print("\nDataset final shape:", final_df.shape)
print("Columnas finales:", final_df.columns.tolist())
print("\nMuestra de los datos finales:")
print(final_df.head())

# Verificar que no hay valores faltantes
print("\nVerificación de valores faltantes en el dataset final:")
print(final_df.isnull().sum())

# Save the final dataset
output_path = 'datasets/final_dataset.csv'
final_df.to_csv(output_path, index=False)

print("\nDataset final guardado en:", output_path)
