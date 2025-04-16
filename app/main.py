from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from app.services.predict_obesity import ObesityPredictor
from app.services.predict_diabetes import DiabetesPredictor
from app.services.predict_hypertension import HypertensionPredictor
import uvicorn

app = FastAPI(
    title="PrevenIA Pro API",
    description="API para predicción de riesgo de obesidad, diabetes e hipertensión usando ensemble de modelos ML",
    version="1.0.0"
)

# Inicializar los predictores (solo una vez)
obesity_predictor = ObesityPredictor()
diabetes_predictor = DiabetesPredictor()
hypertension_predictor = HypertensionPredictor()


class BasePatientData(BaseModel):
    age: int = Field(..., ge=18, le=100,
                     description="Edad del paciente (18-100 años)")
    gender: Literal['male',
                    'female'] = Field(..., description="Género del paciente")
    height: float = Field(..., ge=100, le=250,
                          description="Altura en cm (100-250 cm)")
    weight: float = Field(..., ge=30, le=300,
                          description="Peso en kg (30-300 kg)")
    bmi: float = Field(..., ge=10, le=100,
                       description="Índice de masa corporal")
    physical_activity: Literal['none', 'light', 'moderate', 'frequent'] = Field(
        ..., description="Nivel de actividad física")
    smoker: int = Field(..., ge=0, le=1, description="Fumador (0=No, 1=Sí)")
    alcohol_consumption: Literal['none', 'light', 'moderate', 'heavy'] = Field(
        ..., description="Nivel de consumo de alcohol")
    fasting_glucose: float = Field(
        ..., ge=50, le=500, description="Glucosa en ayunas (mg/dL)")
    systolic_bp: float = Field(
        ..., ge=70, le=250, description="Presión arterial sistólica (mmHg)")

    @validator('bmi')
    def validate_bmi(cls, v, values):
        if 'weight' in values and 'height' in values:
            calculated_bmi = values['weight'] / ((values['height'] / 100) ** 2)
            if abs(v - calculated_bmi) > 1:  # Permitimos una diferencia de ±1
                raise ValueError(
                    "El BMI proporcionado no coincide con el calculado a partir del peso y la altura")
        return v


class ObesityPatientData(BasePatientData):
    family_history_obesity: int = Field(
        ..., ge=0, le=1, description="Historia familiar de obesidad (0=No, 1=Sí)")


class DiabetesPatientData(BasePatientData):
    hba1c: Optional[float] = Field(
        None, ge=3.0, le=15.0, description="Hemoglobina glucosilada (%)")
    family_history_diabetes: int = Field(
        ..., ge=0, le=1, description="Historia familiar de diabetes (0=No, 1=Sí)")
    triglycerides: float = Field(
        ..., ge=0, le=1000, description="Triglicéridos (mg/dL)")


class HypertensionPatientData(BasePatientData):
    diastolic_bp: float = Field(
        ..., ge=40, le=150, description="Presión arterial diastólica (mmHg)")
    family_history_hypertension: int = Field(
        ..., ge=0, le=1, description="Historia familiar de hipertensión (0=No, 1=Sí)")


class CombinedPatientData(BasePatientData):
    # Campos adicionales para diabetes
    hba1c: Optional[float] = Field(None, ge=4.0, le=15.0,
                                   description="Hemoglobina glucosilada (%)")
    family_history_diabetes: int = Field(..., ge=0, le=1,
                                         description="Historia familiar de diabetes (0=No, 1=Sí)")
    triglycerides: float = Field(..., ge=0, le=1000,
                                 description="Triglicéridos (mg/dL)")

    # Campos adicionales para obesidad
    family_history_obesity: int = Field(..., ge=0, le=1,
                                        description="Historia familiar de obesidad (0=No, 1=Sí)")

    # Campos adicionales para hipertensión
    diastolic_bp: float = Field(..., ge=40, le=150,
                                description="Presión arterial diastólica (mmHg)")
    family_history_hypertension: int = Field(
        ..., ge=0, le=1, description="Historia familiar de hipertensión (0=No, 1=Sí)")


@app.post("/predict/obesity")
async def predict_obesity(patient: ObesityPatientData):
    try:
        # Convertir el modelo Pydantic a diccionario
        patient_data = patient.dict()

        # Realizar predicción
        result = obesity_predictor.predict(patient_data)

        return {
            "status": "success",
            "data": {
                "prediction": result['prediction'],
                "probability": f"{result['probability']:.2%}",
                "risk_level": result['risk_level'],
                "recommendations": result['recommendations'],
                "model_probabilities": {
                    k: f"{v:.2%}" for k, v in result['model_probabilities'].items()
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/diabetes")
async def predict_diabetes(patient: DiabetesPatientData):
    try:
        # Convertir el modelo Pydantic a diccionario
        patient_data = patient.dict()

        # Realizar predicción
        result = diabetes_predictor.predict(patient_data)

        return {
            "status": "success",
            "data": {
                "prediction": result['prediction'],
                "probability": f"{result['probability']:.2%}",
                "risk_level": result['risk_level'],
                "recommendations": result['recommendations'],
                "model_probabilities": {
                    k: f"{v:.2%}" for k, v in result['model_probabilities'].items()
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/hypertension")
async def predict_hypertension(patient: HypertensionPatientData):
    try:
        # Convertir el modelo Pydantic a diccionario
        patient_data = patient.dict()

        # Realizar predicción
        result = hypertension_predictor.predict(patient_data)

        return {
            "status": "success",
            "data": {
                "prediction": result['prediction'],
                "probability": f"{result['probability']:.2%}",
                "risk_level": result['risk_level'],
                "recommendations": result['recommendations'],
                "model_probabilities": {
                    k: f"{v:.2%}" for k, v in result['model_probabilities'].items()
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/combined")
async def predict_combined(patient: CombinedPatientData):
    try:
        # Convertir el modelo Pydantic a diccionario
        patient_data = patient.dict()

        # Realizar predicciones individuales
        obesity_result = obesity_predictor.predict(patient_data)
        diabetes_result = diabetes_predictor.predict(patient_data)
        hypertension_result = hypertension_predictor.predict(patient_data)

        # Combinar recomendaciones eliminando duplicados
        all_recommendations = set()
        all_recommendations.update(obesity_result['recommendations'])
        all_recommendations.update(diabetes_result['recommendations'])
        all_recommendations.update(hypertension_result['recommendations'])

        # Determinar el nivel de riesgo general
        risk_levels = {
            'Bajo': 1,
            'Medio': 2,
            'Medio-Alto': 3,
            'Alto': 4
        }

        risk_scores = [
            risk_levels.get(obesity_result['risk_level'], 1),
            risk_levels.get(diabetes_result['risk_level'], 1),
            risk_levels.get(hypertension_result['risk_level'], 1)
        ]

        max_risk_score = max(risk_scores)
        overall_risk_level = {
            1: 'Bajo',
            2: 'Medio',
            3: 'Medio-Alto',
            4: 'Alto'
        }[max_risk_score]

        return {
            "status": "success",
            "data": {
                "obesity": {
                    "prediction": obesity_result['prediction'],
                    "probability": f"{obesity_result['probability']:.2%}",
                    "risk_level": obesity_result['risk_level'],
                    "model_probabilities": {
                        k: f"{v:.2%}" for k, v in obesity_result['model_probabilities'].items()
                    }
                },
                "diabetes": {
                    "prediction": diabetes_result['prediction'],
                    "probability": f"{diabetes_result['probability']:.2%}",
                    "risk_level": diabetes_result['risk_level'],
                    "model_probabilities": {
                        k: f"{v:.2%}" for k, v in diabetes_result['model_probabilities'].items()
                    }
                },
                "hypertension": {
                    "prediction": hypertension_result['prediction'],
                    "probability": f"{hypertension_result['probability']:.2%}",
                    "risk_level": hypertension_result['risk_level'],
                    "model_probabilities": {
                        k: f"{v:.2%}" for k, v in hypertension_result['model_probabilities'].items()
                    }
                },
                "overall_risk_level": overall_risk_level,
                "combined_recommendations": sorted(list(all_recommendations))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "PrevenIA Pro API",
        "version": "1.0.0",
        "endpoints": {
            "/predict/obesity": "POST - Realizar predicción de obesidad",
            "/predict/diabetes": "POST - Realizar predicción de diabetes",
            "/predict/hypertension": "POST - Realizar predicción de hipertensión",
            "/predict/combined": "POST - Realizar predicción combinada de las tres condiciones",
            "/docs": "GET - Documentación interactiva de la API"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
