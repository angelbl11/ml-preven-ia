from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="API de Predicción de Riesgo de Comorbilidades")


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


class ObesityRiskPrediction(BaseModel):
    risk_category: str
    probability: float
    is_obese: bool


class DiabetesRiskPrediction(BaseModel):
    risk_category: str
    probability: float
    is_diabetic: bool


class HypertensionRiskPrediction(BaseModel):
    risk_category: str
    probability: float
    is_hypertensive: bool


# --- Función de utilidad para categorizar riesgo ---


def categorize_risk(probability: float, low_medium_threshold: float = 0.5, medium_high_threshold: float = 0.8) -> str:
    if probability < low_medium_threshold:
        return "Bajo"
    elif probability < medium_high_threshold:
        return "Medio"
    else:
        return "Alto"

# --- Endpoints de la API ---


@app.post("/predict/obesity", response_model=ObesityRiskPrediction, tags=["Obesidad"])
async def predict_obesity(patient_data: PatientObesityData):
    """
    Endpoint para la predicción de riesgo de obesidad.

    Este endpoint recibe datos de un paciente y predice su riesgo de obesidad.
    Utiliza un modelo de machine learning entrenado para realizar la predicción.
    """
    try:
        df_patient = pd.DataFrame([patient_data.dict()])
        features = ['imc', 'ldl', 'trigliceridos',
                    'glucosa_ayunas', 'insulina', 'condicion_genetica']
        # Aplicar el scaler específico para obesidad
        df_patient[features] = scaler_obesity.transform(df_patient[features])
        X_patient = df_patient[features]
        prediction = obesity_model.predict(X_patient)[0]
        obesity_probability = obesity_model.predict_proba(X_patient)[0][1]
        is_obese = bool(prediction)
        risk_category = categorize_risk(obesity_probability)
        return {"risk_category": risk_category,
                "probability": float(obesity_probability),
                "is_obese": is_obese
                }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la solicitud de obesidad: {e}")


@app.post("/predict/diabetes", response_model=DiabetesRiskPrediction, tags=["Diabetes"])
async def predict_diabetes(patient_data: PatientDiabetesData):
    """
    Endpoint para la predicción de riesgo de diabetes.

    Este endpoint recibe datos de un paciente y predice su riesgo de diabetes.
    Utiliza un modelo de machine learning entrenado para realizar la predicción.
    """
    try:
        df_patient = pd.DataFrame([patient_data.dict()])
        features = ['glucosa_ayunas', 'insulina',
                    'hba1c', 'condicion_genetica']
        # Aplicar el scaler específico para diabetes
        df_patient[features] = scaler_diabetes.transform(df_patient[features])
        X_patient = df_patient[features]
        prediction = diabetes_model.predict(X_patient)[0]
        diabetes_probability = diabetes_model.predict_proba(X_patient)[0][1]
        is_diabetic = bool(prediction)
        risk_category = categorize_risk(diabetes_probability)
        return {"risk_category": risk_category,
                "probability": float(diabetes_probability),
                "is_diabetic": is_diabetic
                }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la solicitud de diabetes: {e}")


@app.post("/predict/hipertension", response_model=HypertensionRiskPrediction, tags=["Hipertension"])
async def predict_hipertension(patient_data: PatientHypertensionData):
    """
    Endpoint para la predicción de riesgo de hipertensión.

    Este endpoint recibe datos de un paciente y predice su riesgo de hipertensión.
    Utiliza un modelo de machine learning entrenado para realizar la predicción.
    """
    try:
        df_patient = pd.DataFrame([patient_data.dict()])
        features = ['presion_arterial_sistolica', 'presion_arterial_diastolica',
                    'creatinina', 'ldl', 'condicion_genetica', 'edad']
        # Aplicar el scaler específico para hipertension
        df_patient[features] = scaler_hipertension.transform(
            df_patient[features])
        X_patient = df_patient[features]
        prediction = hipertension_model.predict(X_patient)[0]
        hipertension_probability = hipertension_model.predict_proba(X_patient)[
            0][1]
        is_hypertensive = bool(prediction)
        risk_category = categorize_risk(hipertension_probability)
        return {"risk_category": risk_category,
                "probability": float(hipertension_probability),
                "is_hypertensive": is_hypertensive
                }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la solicitud de hipertensión: {e}")
