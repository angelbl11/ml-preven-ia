from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(
    title="API de Predicción de Riesgo de Comorbilidades",
    description="API para predecir el riesgo de obesidad, diabetes e hipertensión."
)

# --- Cargar modelos y scalers específicos ---
try:
    with open("models/obesity_model.pkl", "rb") as f:
        obesity_model = pickle.load(f)
    with open("models/scaler_obesity.pkl", "rb") as f:
        scaler_obesity = pickle.load(f)
    with open("models/diabetes_model.pkl", "rb") as f:
        diabetes_model = pickle.load(f)
    with open("models/scaler_diabetes.pkl", "rb") as f:
        scaler_diabetes = pickle.load(f)
    with open("models/hipertension_model.pkl", "rb") as f:
        hipertension_model = pickle.load(f)
    with open("models/scaler_hipertension.pkl", "rb") as f:
        scaler_hipertension = pickle.load(f)
    print("Modelos y scalers cargados exitosamente.")
except Exception as e:
    raise Exception(f"Error al cargar modelos y scalers: {e}")

# --- Definición de modelos de datos con Pydantic ---


class PatientObesityData(BaseModel):
    imc: float
    ldl: float
    trigliceridos: float
    glucosa_ayunas: float
    insulina: float
    condicion_genetica: int


class PatientDiabetesData(BaseModel):
    glucosa_ayunas: float
    insulina: float
    hba1c: float
    condicion_genetica: int


class PatientHypertensionData(BaseModel):
    presion_arterial_sistolica: float
    presion_arterial_diastolica: float
    creatinina: float
    ldl: float
    condicion_genetica: int
    edad: int


class RiskPrediction(BaseModel):
    risk_category: str
    probability: float

# --- Función de utilidad para categorizar riesgo ---


def categorize_risk(probability: float, low_medium_threshold: float = 0.5, medium_high_threshold: float = 0.8) -> str:
    if probability < low_medium_threshold:
        return "Bajo"
    elif probability < medium_high_threshold:
        return "Medio"
    else:
        return "Alto"

# --- Endpoints de la API ---


@app.post("/predict/obesity", response_model=RiskPrediction, tags=["Obesidad"])
async def predict_obesity(patient_data: PatientObesityData):
    try:
        df_patient = pd.DataFrame([patient_data.dict()])
        features = ['imc', 'ldl', 'trigliceridos',
                    'glucosa_ayunas', 'insulina', 'condicion_genetica']
        # Aplicar el scaler específico para obesidad
        df_patient[features] = scaler_obesity.transform(df_patient[features])
        print("DataFrame escalado para obesidad:", df_patient[features])
        X_patient = df_patient[features]
        probability = obesity_model.predict_proba(X_patient)[:, 1][0]
        risk_category = categorize_risk(probability)
        return {"risk_category": risk_category, "probability": float(probability)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la solicitud de obesidad: {e}")


@app.post("/predict/diabetes", response_model=RiskPrediction, tags=["Diabetes"])
async def predict_diabetes(patient_data: PatientDiabetesData):
    try:
        df_patient = pd.DataFrame([patient_data.dict()])
        features = ['glucosa_ayunas', 'insulina',
                    'hba1c', 'condicion_genetica']
        # Aplicar el scaler específico para diabetes
        df_patient[features] = scaler_diabetes.transform(df_patient[features])
        X_patient = df_patient[features]
        probability = diabetes_model.predict_proba(X_patient)[:, 1][0]
        risk_category = categorize_risk(probability)
        return {"risk_category": risk_category, "probability": float(probability)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la solicitud de diabetes: {e}")


@app.post("/predict/hipertension", response_model=RiskPrediction, tags=["Hipertension"])
async def predict_hipertension(patient_data: PatientHypertensionData):
    try:
        df_patient = pd.DataFrame([patient_data.dict()])
        features = ['presion_arterial_sistolica', 'presion_arterial_diastolica',
                    'creatinina', 'ldl', 'condicion_genetica', 'edad']
        # Aplicar el scaler específico para hipertensión
        df_patient[features] = scaler_hipertension.transform(
            df_patient[features])
        X_patient = df_patient[features]
        probability = hipertension_model.predict_proba(X_patient)[:, 1][0]
        risk_category = categorize_risk(probability)
        return {"risk_category": risk_category, "probability": float(probability)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la solicitud de hipertensión: {e}")
