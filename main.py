from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="API de Predicción de Riesgo de Comorbilidades")

# --- DEFINICIÓN DE FEATURES (DEBEN COINCIDIR EXACTAMENTE CON EL NOTEBOOK) ---
features_obesity = [
    'imc_cat_sobrepeso',
    'imc_cat_obesidad_g1',
    'imc_cat_obesidad_g2',
    'imc_cat_obesidad_g3',
    'ldl',
    'trigliceridos',
    'condicion_genetica',
    'genero',
    'imc_edad_interaccion'
]

features_diabetes = [
    'glucosa_ayunas',
    'hba1c',
    'condicion_genetica',
    'genero',
    'glucosa_hba1c_interaccion',
    'imc_cat_sobrepeso',
    'imc_cat_obesidad_g1',
    'imc_cat_obesidad_g2',
    'imc_cat_obesidad_g3',
    'edad_cat_18-29',
    'edad_cat_30-39',
    'edad_cat_40-49',
    'edad_cat_50+'
]

features_hipertension = [
    'presion_arterial_sistolica',
    'presion_arterial_diastolica',
    'creatinina',
    'condicion_genetica',
    'genero',
    'edad_cat_50+',
    'imc_cat_sobrepeso',
    'imc_cat_obesidad_g1',
    'imc_cat_obesidad_g2',
    'imc_cat_obesidad_g3',
    'ratio_presion_arterial',
    'imc_edad_interaccion'
]


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
    imc: float = Field(..., description="Índice de Masa Corporal")
    ldl: float = Field(..., description="Colesterol LDL")
    trigliceridos: float = Field(..., description="Nivel de Triglicéridos")
    condicion_genetica: bool = Field(...,
                                     description="Condición Genética (true/false)")
    genero: str = Field(..., description="Género ('male'/'female')")
    edad: int = Field(..., description="Edad del paciente")


class PatientDiabetesData(BaseModel):
    glucosa_ayunas: float = Field(..., description="Glucosa en Ayunas")
    hba1c: float = Field(..., description="Hemoglobina Glicosilada (HbA1c)")
    condicion_genetica: bool = Field(...,
                                     description="Condición Genética (true/false)")
    genero: str = Field(..., description="Género ('male'/'female')")
    edad: int = Field(..., description="Edad del paciente")
    imc: float = Field(..., description="Índice de Masa Corporal")


class PatientHypertensionData(BaseModel):
    presion_arterial_sistolica: float = Field(
        ..., description="Presión Arterial Sistólica")
    presion_arterial_diastolica: float = Field(
        ..., description="Presión Arterial Diastólica")
    creatinina: float = Field(..., description="Nivel de Creatinina")
    ldl: float = Field(..., description="Colesterol LDL")
    condicion_genetica: bool = Field(...,
                                     description="Condición Genética (true/false)")
    genero: str = Field(..., description="Género ('male'/'female')")
    edad: int = Field(..., description="Edad del paciente")
    imc: float = Field(..., description="Índice de Masa Corporal")


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

# --- Middleware para manejar ValidationErrors ---


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# --- Endpoints de la API ---


@app.post("/predict/obesity", response_model=ObesityRiskPrediction, tags=["Obesidad"])
async def predict_obesity(patient_data: PatientObesityData):
    """
    Endpoint para la predicción de riesgo de obesidad.
    """
    try:
        patient_dict = patient_data.dict()

        patient_dict['condicion_genetica'] = 1 if patient_dict['condicion_genetica'] else 0
        patient_dict['genero'] = 1 if patient_dict['genero'] == 'male' else 0

        df_patient = pd.DataFrame([patient_dict])

        bins_imc = [0, 25, 30, 35, 40, 100]
        labels_imc = ['normal', 'sobrepeso',
                      'obesidad_g1', 'obesidad_g2', 'obesidad_g3']
        df_patient['imc_categoria'] = pd.cut(
            df_patient['imc'], bins=bins_imc, labels=labels_imc, right=False)
        df_patient = pd.get_dummies(
            df_patient, columns=['imc_categoria'], prefix='imc_cat', drop_first=True)

        for cat_feature in ['imc_cat_sobrepeso', 'imc_cat_obesidad_g1', 'imc_cat_obesidad_g2', 'imc_cat_obesidad_g3']:
            if cat_feature not in df_patient.columns:
                df_patient[cat_feature] = 0

        df_patient['imc_edad_interaccion'] = df_patient['imc'] * \
            df_patient['edad']

        df_patient_scaled = df_patient[features_obesity].copy()
        numeric_features_obesity_scaler = [
            'ldl', 'trigliceridos', 'genero', 'imc_edad_interaccion']
        df_patient_scaled[numeric_features_obesity_scaler] = scaler_obesity.transform(
            df_patient_scaled[numeric_features_obesity_scaler])
        X_patient = df_patient_scaled[features_obesity]

        prediction = obesity_model.predict(X_patient)[0]
        obesity_probability = obesity_model.predict_proba(X_patient)[0][1]
        is_obese = bool(prediction)
        risk_category = categorize_risk(obesity_probability)
        return {
            "risk_category": risk_category,
            "probability": round(float(obesity_probability), 4),
            "is_obese": is_obese
        }
    except ValidationError as ve:
        raise ve
    except ValueError as ve:
        error_message = f"Error en los datos de entrada: {str(ve)}"
        raise HTTPException(status_code=400, detail=error_message)
    except TypeError as te:
        error_message = f"Error de tipo en el procesamiento de datos: {str(te)}"
        raise HTTPException(status_code=400, detail=error_message)
    except Exception as e:
        error_message = f"Error interno del servidor al procesar la solicitud de obesidad: {e.__class__.__name__} - {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/predict/diabetes", response_model=DiabetesRiskPrediction, tags=["Diabetes"])
async def predict_diabetes(patient_data: PatientDiabetesData):
    """
    Endpoint para la predicción de riesgo de diabetes.
    """
    try:
        patient_dict = patient_data.dict()

        patient_dict['condicion_genetica'] = 1 if patient_dict['condicion_genetica'] else 0
        patient_dict['genero'] = 1 if patient_dict['genero'] == 'male' else 0

        df_patient = pd.DataFrame([patient_dict])

        # --- INGENIERÍA DE CARACTERÍSTICAS PARA DIABETES (REPLICAR DEL NOTEBOOK) ---
        df_patient['glucosa_hba1c_interaccion'] = df_patient['glucosa_ayunas'] * \
            df_patient['hba1c']

        bins_imc = [0, 25, 30, 35, 40, 100]
        labels_imc = ['normal', 'sobrepeso',
                      'obesidad_g1', 'obesidad_g2', 'obesidad_g3']
        df_patient['imc_categoria'] = pd.cut(
            df_patient['imc'], bins=bins_imc, labels=labels_imc, right=False)
        df_patient = pd.get_dummies(
            df_patient, columns=['imc_categoria'], prefix='imc_cat', drop_first=True)

        bins_edad = [0, 30, 40, 50, 120]
        labels_edad = ['18-29', '30-39', '40-49', '50+']
        df_patient['edad_categoria'] = pd.cut(
            df_patient['edad'], bins=bins_edad, labels=labels_edad, right=False)
        df_patient = pd.get_dummies(
            df_patient, columns=['edad_categoria'], prefix='edad_cat', drop_first=True)

        # Asegurar que las columnas categóricas IMC y EDAD estén presentes
        for cat_feature in ['imc_cat_sobrepeso', 'imc_cat_obesidad_g1', 'imc_cat_obesidad_g2', 'imc_cat_obesidad_g3',
                            'edad_cat_18-29', 'edad_cat_30-39', 'edad_cat_40-49', 'edad_cat_50+']:
            if cat_feature not in df_patient.columns:
                df_patient[cat_feature] = 0

        df_patient_scaled = df_patient[features_diabetes].copy()
        numeric_features_diabetes_scaler = [
            'glucosa_ayunas', 'hba1c', 'genero', 'glucosa_hba1c_interaccion']
        df_patient_scaled[numeric_features_diabetes_scaler] = scaler_diabetes.transform(
            df_patient_scaled[numeric_features_diabetes_scaler])
        X_patient = df_patient_scaled[features_diabetes]

        prediction = diabetes_model.predict(X_patient)[0]
        diabetes_probability = diabetes_model.predict_proba(X_patient)[0][1]
        is_diabetic = bool(prediction)
        risk_category = categorize_risk(diabetes_probability)
        return {
            "risk_category": risk_category,
            "probability": round(float(diabetes_probability), 4),
            "is_diabetic": is_diabetic
        }
    except ValidationError as ve:
        raise ve
    except ValueError as ve:
        error_message = f"Error en los datos de entrada: {str(ve)}"
        raise HTTPException(status_code=400, detail=error_message)
    except TypeError as te:
        error_message = f"Error de tipo en el procesamiento de datos: {str(te)}"
        raise HTTPException(status_code=400, detail=error_message)
    except Exception as e:
        error_message = f"Error interno del servidor al procesar la solicitud de diabetes: {e.__class__.__name__} - {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/predict/hipertension", response_model=HypertensionRiskPrediction, tags=["Hipertension"])
async def predict_hipertension(patient_data: PatientHypertensionData):
    """
    Endpoint para la predicción de riesgo de hipertensión.
    """
    try:
        print("----- Recibiendo solicitud para /predict/hipertension -----")
        patient_dict = patient_data.dict()
        # Convertir 'genero' y 'condicion_genetica' a numérico
        patient_dict['genero'] = 1 if patient_dict['genero'] == 'male' else 0
        patient_dict['condicion_genetica'] = 1 if patient_dict['condicion_genetica'] else 0
        print("patient_dict:", patient_dict)

        df_patient = pd.DataFrame([patient_dict])
        print("df_patient columns BEFORE feature engineering:", df_patient.columns)

        # Ingeniería de características para IMC
        bins_imc = [0, 25, 30, 35, 40, 100]
        labels_imc = ['normal', 'sobrepeso',
                      'obesidad_g1', 'obesidad_g2', 'obesidad_g3']
        df_patient['imc_categoria'] = pd.cut(
            df_patient['imc'], bins=bins_imc, labels=labels_imc, right=False)
        df_patient = pd.get_dummies(
            df_patient, columns=['imc_categoria'], prefix='imc_cat', drop_first=True)

        # Ingeniería de características para EDAD
        bins_edad = [0, 50, 120]
        labels_edad = ['<50', '50+']
        df_patient['edad_categoria'] = pd.cut(
            df_patient['edad'], bins=bins_edad, labels=labels_edad, right=False)
        df_patient = pd.get_dummies(
            df_patient, columns=['edad_categoria'], prefix='edad_cat', drop_first=True)

        # No se aplica one-hot encoding a 'genero', se mantiene numérico.
        # Ingeniería de características adicionales
        df_patient['ratio_presion_arterial'] = df_patient['presion_arterial_sistolica'] / \
            df_patient['presion_arterial_diastolica']
        df_patient['imc_edad_interaccion'] = df_patient['imc'] * \
            df_patient['edad']
        print("df_patient columns AFTER feature engineering:", df_patient.columns)

        # Asegurar que las columnas categóricas estén presentes
        for cat_feature in ['imc_cat_sobrepeso', 'imc_cat_obesidad_g1', 'imc_cat_obesidad_g2', 'imc_cat_obesidad_g3', 'edad_cat_50+']:
            if cat_feature not in df_patient.columns:
                df_patient[cat_feature] = 0

        # Seleccionar las features definidas globalmente
        df_patient_scaled = df_patient[features_hipertension].copy()
        # Actualizar la lista de columnas a escalar (se elimina 'condicion_genetica')
        numeric_features_hipertension_scaler = [
            'presion_arterial_sistolica',
            'presion_arterial_diastolica',
            'creatinina',
            'genero',
            'ratio_presion_arterial',
            'imc_edad_interaccion'
        ]
        features_to_scale = [
            feature for feature in numeric_features_hipertension_scaler if feature in df_patient_scaled.columns]
        df_patient_scaled[features_to_scale] = scaler_hipertension.transform(
            df_patient_scaled[features_to_scale])
        X_patient = df_patient_scaled[features_hipertension]

        prediction = hipertension_model.predict(X_patient)[0]
        hipertension_probability = hipertension_model.predict_proba(X_patient)[
            0][1]
        is_hypertensive = bool(prediction)
        risk_category = categorize_risk(hipertension_probability)
        return {
            "risk_category": risk_category,
            "probability": round(float(hipertension_probability), 4),
            "is_hypertensive": is_hypertensive
        }
    except Exception as e:
        error_message = f"Error interno del servidor al procesar la solicitud de hipertension: {e.__class__.__name__} - {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)
