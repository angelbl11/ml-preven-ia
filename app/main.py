from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, List, Dict
from app.services.predict_obesity import ObesityPredictor
from app.services.predict_diabetes import DiabetesPredictor
from app.services.predict_hypertension import HypertensionPredictor
import uvicorn
from enum import Enum

app = FastAPI(
    title="PrevenIA Pro API",
    description="""
    API para predicción de riesgo de obesidad, diabetes e hipertensión usando ensemble de modelos ML.
    
    ## Características principales
    
    - Predicción individual de riesgo de obesidad, diabetes e hipertensión
    - Predicción combinada de los tres riesgos
    - Recomendaciones personalizadas basadas en el perfil del paciente
    - Análisis de probabilidades por modelo individual
    
    ## Endpoints disponibles
    
    - `/predict/obesity`: Predicción de riesgo de obesidad
    - `/predict/diabetes`: Predicción de riesgo de diabetes
    - `/predict/hypertension`: Predicción de riesgo de hipertensión
    - `/predict/combined`: Predicción combinada de los tres riesgos
    
    ## Modelos utilizados
    
    - Random Forest
    - LightGBM
    - XGBoost
    - Modelos basados en criterios clínicos
    
    ## Ejemplo de uso
    
    ```python
    import requests
    
    # Datos del paciente
    patient_data = {
        "age": 45,
        "gender": "male",
        "height": 175,
        "weight": 85,
        "bmi": 27.8,
        "physical_activity": "moderate",
        "smoker": 0,
        "alcohol_consumption": "light",
        "fasting_glucose": 95,
        "systolic_bp": 130,
        "family_history_obesity": 1
    }
    
    # Realizar predicción
    response = requests.post("http://api.example.com/predict/obesity", json=patient_data)
    result = response.json()
    ```
    """,
    version="1.0.0",
    contact={
        "name": "Equipo PrevenIA",
        "email": "contacto@prevenia.com",
        "url": "https://prevenia.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
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


class ObesityPrediction(str, Enum):
    UNDERWEIGHT = "Bajo peso"
    NORMAL = "Peso normal"
    OVERWEIGHT = "Sobrepeso"
    OBESITY_1 = "Obesidad Tipo 1"
    OBESITY_2 = "Obesidad Tipo 2"
    OBESITY_3 = "Obesidad Tipo 3"


class DiabetesPrediction(str, Enum):
    NORMAL = "Normal"
    PREDIABETES = "Prediabetes"
    DIABETES = "Diabetes"


class HypertensionPrediction(str, Enum):
    NORMAL = "Normal"
    ELEVATED = "Elevada"
    HYPERTENSION_1 = "Hipertensión Etapa 1"
    HYPERTENSION_2 = "Hipertensión Etapa 2"
    HYPERTENSIVE_CRISIS = "Crisis Hipertensiva"


class RiskLevel(str, Enum):
    VERY_LOW = "Muy Bajo"
    LOW = "Bajo"
    MODERATE = "Moderado"
    HIGH = "Alto"
    VERY_HIGH = "Muy Alto"


class ModelProbabilities(BaseModel):
    random_forest: str
    lightgbm: str
    xgboost: str
    bmi_based: str
    risk_factors: str


class ObesityResponse(BaseModel):
    prediction: ObesityPrediction
    probability: str
    risk_level: RiskLevel
    recommendations: List[str]
    model_probabilities: ModelProbabilities


class DiabetesResponse(BaseModel):
    prediction: DiabetesPrediction
    probability: str
    risk_level: RiskLevel
    recommendations: List[str]
    model_probabilities: ModelProbabilities


class HypertensionResponse(BaseModel):
    prediction: HypertensionPrediction
    probability: str
    risk_level: RiskLevel
    recommendations: List[str]
    model_probabilities: ModelProbabilities


class CombinedResponse(BaseModel):
    obesity: Dict[str, str | List[str] | Dict[str, str]]
    diabetes: Dict[str, str | List[str] | Dict[str, str]]
    hypertension: Dict[str, str | List[str] | Dict[str, str]]
    overall_risk_level: RiskLevel
    combined_recommendations: List[str]


class APIResponse(BaseModel):
    status: str
    data: Dict


class ObesityAPIResponse(BaseModel):
    status: str
    data: ObesityResponse


class DiabetesAPIResponse(BaseModel):
    status: str
    data: DiabetesResponse


class HypertensionAPIResponse(BaseModel):
    status: str
    data: HypertensionResponse


class CombinedAPIResponse(BaseModel):
    status: str
    data: CombinedResponse


@app.post("/predict/obesity",
          response_model=ObesityAPIResponse,
          responses={
              200: {
                  "description": "Predicción exitosa",
                  "content": {
                      "application/json": {
                          "example": {
                              "status": "success",
                              "data": {
                                  "prediction": "Sobrepeso",
                                  "probability": "70.00%",
                                  "risk_level": "Moderado",
                                  "recommendations": [
                                      "Consulta con nutricionista",
                                      "Incrementar actividad física gradualmente",
                                      "Modificación de hábitos alimenticios",
                                      "Control regular de peso y medidas",
                                      "Evaluación trimestral de progreso"
                                  ],
                                  "model_probabilities": {
                                      "random_forest": "65.00%",
                                      "lightgbm": "72.00%",
                                      "xgboost": "68.00%",
                                      "bmi_based": "75.00%",
                                      "risk_factors": "60.00%"
                                  }
                              }
                          }
                      }
                  }
              },
              500: {
                  "description": "Error interno del servidor",
                  "content": {
                      "application/json": {
                          "example": {
                              "detail": "Error al procesar la predicción"
                          }
                      }
                  }
              }
          },
          summary="Predicción de riesgo de obesidad",
          description="""
    Realiza una predicción del riesgo de obesidad basada en los datos del paciente.
    
    ## Parámetros de entrada
    - Datos demográficos (edad, género)
    - Medidas antropométricas (altura, peso, IMC)
    - Factores de estilo de vida (actividad física, tabaquismo, consumo de alcohol)
    - Factores de riesgo (historia familiar, glucosa, presión arterial)
    
    ## Respuesta
    - Predicción de categoría de IMC (Bajo peso, Peso normal, Sobrepeso, Obesidad Tipo 1, Obesidad Tipo 2, Obesidad Tipo 3)
    - Probabilidad de riesgo
    - Nivel de riesgo (Muy Bajo, Bajo, Moderado, Alto, Muy Alto)
    - Recomendaciones personalizadas
    - Probabilidades por modelo individual
    """,
          tags=["Predicción de Riesgo"]
          )
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


@app.post("/predict/diabetes",
          response_model=DiabetesAPIResponse,
          responses={
              200: {
                  "description": "Predicción exitosa",
                  "content": {
                      "application/json": {
                          "example": {
                              "status": "success",
                              "data": {
                                  "prediction": "Prediabetes",
                                  "probability": "45.00%",
                                  "risk_level": "Moderado",
                                  "recommendations": [
                                      "Control de glucosa regular",
                                      "Modificación de dieta",
                                      "Ejercicio regular",
                                      "Control de peso",
                                      "Evaluación médica periódica"
                                  ],
                                  "model_probabilities": {
                                      "random_forest": "42.00%",
                                      "lightgbm": "48.00%",
                                      "xgboost": "45.00%",
                                      "bmi_based": "50.00%",
                                      "risk_factors": "40.00%"
                                  }
                              }
                          }
                      }
                  }
              },
              500: {
                  "description": "Error interno del servidor",
                  "content": {
                      "application/json": {
                          "example": {
                              "detail": "Error al procesar la predicción"
                          }
                      }
                  }
              }
          },
          summary="Predicción de riesgo de diabetes",
          description="""
    Realiza una predicción del riesgo de diabetes basada en los datos del paciente.
    
    ## Parámetros de entrada
    - Datos demográficos (edad, género)
    - Medidas antropométricas (altura, peso, IMC)
    - Factores de estilo de vida (actividad física, tabaquismo, consumo de alcohol)
    - Factores de riesgo (historia familiar, HbA1c, glucosa, triglicéridos)
    
    ## Respuesta
    - Predicción de riesgo de diabetes (Normal, Prediabetes, Diabetes)
    - Probabilidad de riesgo
    - Nivel de riesgo (Muy Bajo, Bajo, Moderado, Alto, Muy Alto)
    - Recomendaciones personalizadas
    - Probabilidades por modelo individual
    """,
          tags=["Predicción de Riesgo"]
          )
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


@app.post("/predict/hypertension",
          response_model=HypertensionAPIResponse,
          responses={
              200: {
                  "description": "Predicción exitosa",
                  "content": {
                      "application/json": {
                          "example": {
                              "status": "success",
                              "data": {
                                  "prediction": "Hipertensión Etapa 1",
                                  "probability": "65.00%",
                                  "risk_level": "Moderado",
                                  "recommendations": [
                                      "Control de presión arterial regular",
                                      "Reducción de sodio en dieta",
                                      "Ejercicio regular",
                                      "Control de peso",
                                      "Evaluación médica periódica"
                                  ],
                                  "model_probabilities": {
                                      "random_forest": "62.00%",
                                      "lightgbm": "68.00%",
                                      "xgboost": "65.00%",
                                      "bmi_based": "70.00%",
                                      "risk_factors": "60.00%"
                                  }
                              }
                          }
                      }
                  }
              },
              500: {
                  "description": "Error interno del servidor",
                  "content": {
                      "application/json": {
                          "example": {
                              "detail": "Error al procesar la predicción"
                          }
                      }
                  }
              }
          },
          summary="Predicción de riesgo de hipertensión",
          description="""
    Realiza una predicción del riesgo de hipertensión basada en los datos del paciente.
    
    ## Parámetros de entrada
    - Datos demográficos (edad, género)
    - Medidas antropométricas (altura, peso, IMC)
    - Factores de estilo de vida (actividad física, tabaquismo, consumo de alcohol)
    - Factores de riesgo (historia familiar, presión arterial)
    
    ## Respuesta
    - Predicción de categoría de presión arterial (Normal, Elevada, Hipertensión Etapa 1, Hipertensión Etapa 2, Crisis Hipertensiva)
    - Probabilidad de riesgo
    - Nivel de riesgo (Muy Bajo, Bajo, Moderado, Alto, Muy Alto)
    - Recomendaciones personalizadas
    - Probabilidades por modelo individual
    """,
          tags=["Predicción de Riesgo"]
          )
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


@app.post("/predict/combined",
          response_model=CombinedAPIResponse,
          responses={
              200: {
                  "description": "Predicción exitosa",
                  "content": {
                      "application/json": {
                          "example": {
                              "status": "success",
                              "data": {
                                  "obesity": {
                                      "prediction": "Sobrepeso",
                                      "probability": "70.00%",
                                      "risk_level": "Moderado",
                                      "model_probabilities": {
                                          "random_forest": "65.00%",
                                          "lightgbm": "72.00%",
                                          "xgboost": "68.00%",
                                          "bmi_based": "75.00%",
                                          "risk_factors": "60.00%"
                                      }
                                  },
                                  "diabetes": {
                                      "prediction": "Prediabetes",
                                      "probability": "45.00%",
                                      "risk_level": "Moderado",
                                      "model_probabilities": {
                                          "random_forest": "42.00%",
                                          "lightgbm": "48.00%",
                                          "xgboost": "45.00%",
                                          "bmi_based": "50.00%",
                                          "risk_factors": "40.00%"
                                      }
                                  },
                                  "hypertension": {
                                      "prediction": "Hipertensión Etapa 1",
                                      "probability": "65.00%",
                                      "risk_level": "Moderado",
                                      "model_probabilities": {
                                          "random_forest": "62.00%",
                                          "lightgbm": "68.00%",
                                          "xgboost": "65.00%",
                                          "bmi_based": "70.00%",
                                          "risk_factors": "60.00%"
                                      }
                                  },
                                  "overall_risk_level": "Moderado",
                                  "combined_recommendations": [
                                      "Control de peso",
                                      "Ejercicio regular",
                                      "Dieta balanceada",
                                      "Control de presión arterial",
                                      "Control de glucosa",
                                      "Evaluación médica periódica"
                                  ]
                              }
                          }
                      }
                  }
              },
              500: {
                  "description": "Error interno del servidor",
                  "content": {
                      "application/json": {
                          "example": {
                              "detail": "Error al procesar la predicción"
                          }
                      }
                  }
              }
          },
          summary="Predicción combinada de riesgos",
          description="""
    Realiza una predicción combinada de los riesgos de obesidad, diabetes e hipertensión.
    
    ## Parámetros de entrada
    - Datos demográficos (edad, género)
    - Medidas antropométricas (altura, peso, IMC)
    - Factores de estilo de vida (actividad física, tabaquismo, consumo de alcohol)
    - Factores de riesgo (historia familiar, HbA1c, glucosa, triglicéridos, presión arterial)
    
    ## Respuesta
    - Predicciones individuales para cada condición
    - Probabilidades de riesgo
    - Niveles de riesgo
    - Recomendaciones personalizadas combinadas
    - Probabilidades por modelo individual
    - Nivel de riesgo general
    """,
          tags=["Predicción de Riesgo"]
          )
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
