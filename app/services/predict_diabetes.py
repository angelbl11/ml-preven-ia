import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import json

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


class DiabetesPredictor:
    def __init__(self):
        # Cargar modelos usando rutas absolutas
        models_dir = os.path.join(BASE_DIR, 'app', 'models', 'diabetes')
        self.rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
        self.lgb_model = joblib.load(os.path.join(models_dir, 'lgb_model.pkl'))
        self.xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))

        # Cargar configuración del modelo
        try:
            with open(os.path.join(models_dir, 'model_config.json'), 'r') as f:
                config = json.load(f)
                self.weights = np.array(config['weights'])
                self.threshold = config['threshold']
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print(
                "Warning: No se pudo cargar la configuración del modelo, usando valores por defecto")
            self.weights = np.array([0.33, 0.33, 0.34])  # RF, LGB, XGB
            self.threshold = 0.5

        # Inicializar transformadores
        self.le = LabelEncoder()
        self.scaler = StandardScaler()

        # Definir rangos para discretización basados en criterios clínicos
        # Normal, Pre-diabetes, Diabetes
        self.hba1c_bins = [0, 5.7, 6.5, float('inf')]
        # Normal, Pre-diabetes, Diabetes
        self.glucose_bins = [0, 100, 126, float('inf')]
        # Normal, Sobrepeso, Obesidad
        self.bmi_bins = [0, 25, 30, float('inf')]
        self.age_bins = [0, 30, 50, float('inf')]  # Joven, Adulto, Mayor
        # Normal, Pre-hipertensión, Hipertensión
        self.bp_bins = [0, 120, 140, float('inf')]
        # Normal, Límite alto, Alto
        self.triglycerides_bins = [0, 150, 200, float('inf')]

    def _safe_transform(self, value, feature_name, bins):
        """Discretiza un valor de forma segura"""
        try:
            value = float(value)
            cat = pd.cut([value], bins=bins, labels=False)[0]
            return 1 if pd.isna(cat) else int(cat)
        except (ValueError, TypeError):
            print(
                f"Warning: Error al transformar {feature_name}, usando valor por defecto")
            return 1

    def _preprocess_input(self, data):
        """
        Preprocesa los datos de entrada para que coincidan con el formato del modelo.

        Parameters:
        -----------
        data : dict
            Diccionario con los datos del paciente

        Returns:
        --------
        pd.DataFrame
            DataFrame con las features procesadas
        """
        # Crear DataFrame con una fila
        df = pd.DataFrame([data])

        # Imputar HbA1c si no está disponible usando un modelo simple basado en glucosa y otros factores
        if 'hba1c' not in data or pd.isna(data['hba1c']):
            # Modelo de imputación simple basado en glucosa y otros factores
            glucose = float(data.get('fasting_glucose', 100))
            bmi = float(data.get('bmi', 25))
            age = float(data.get('age', 45))

            # Fórmula de estimación basada en correlaciones clínicas
            # Fórmula básica de glucosa a HbA1c
            estimated_hba1c = (glucose / 28.7) + 2.15
            # Ajustes basados en otros factores
            if bmi > 30:
                estimated_hba1c += 0.3
            if age > 45:
                estimated_hba1c += 0.2
            if data.get('family_history_diabetes', 0) == 1:
                estimated_hba1c += 0.2

            # Mantener dentro de rangos razonables
            data['hba1c'] = max(4.0, min(15.0, estimated_hba1c))
            print(f"DEBUG - HbA1c imputado: {data['hba1c']}")

        # Discretizar variables numéricas de forma segura
        df['hba1c_cat'] = self._safe_transform(
            data['hba1c'], 'hba1c', self.hba1c_bins)
        df['glucose_cat'] = self._safe_transform(
            data['fasting_glucose'], 'glucose', self.glucose_bins)
        df['bmi_cat'] = self._safe_transform(data['bmi'], 'bmi', self.bmi_bins)
        df['age_cat'] = self._safe_transform(data['age'], 'age', self.age_bins)
        df['bp_cat'] = self._safe_transform(
            data['systolic_bp'], 'bp', self.bp_bins)
        df['triglycerides_cat'] = self._safe_transform(
            data['triglycerides'], 'triglycerides', self.triglycerides_bins)

        # Codificar variables categóricas
        categorical_mapping = {
            'gender': {'male': 0, 'female': 1},
            'physical_activity': {'none': 0, 'light': 1, 'moderate': 2, 'frequent': 3},
            'alcohol_consumption': {'none': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
        }

        # Aplicar mapeo categórico
        for feature, mapping in categorical_mapping.items():
            df[f'{feature}_encoded'] = df[feature].map(mapping).fillna(0)

        # Asegurar que variables binarias son numéricas
        df['smoker'] = pd.to_numeric(df['smoker'], errors='coerce').fillna(0)
        df['family_history_diabetes'] = pd.to_numeric(
            df['family_history_diabetes'], errors='coerce').fillna(0)

        # Seleccionar y ordenar features
        features = [
            'hba1c_cat',     # HbA1c como feature principal
            'glucose_cat',    # Glucosa como segundo feature principal
            'bmi_cat',
            'age_cat',
            'bp_cat',
            'triglycerides_cat',
            'gender_encoded',
            'physical_activity_encoded',
            'alcohol_consumption_encoded',
            'smoker',
            'family_history_diabetes'
        ]

        # Asegurar que no hay valores nulos
        for feature in features:
            if df[feature].isnull().any():
                df[feature] = df[feature].fillna(0)

        # Escalar features numéricas
        numeric_features = ['hba1c_cat', 'glucose_cat',
                            'bmi_cat', 'age_cat', 'bp_cat', 'triglycerides_cat']
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])

        return df[features]

    def predict(self, data):
        """
        Realiza una predicción de diabetes usando el ensemble de modelos.

        Parameters:
        -----------
        data : dict
            Diccionario con los datos del paciente:
            - age: int
            - gender: str ('male' o 'female')
            - height: float (cm)
            - weight: float (kg)
            - bmi: float
            - hba1c: float (%) [opcional]
            - physical_activity: str ('none', 'light', 'moderate', 'frequent')
            - smoker: int (0 o 1)
            - alcohol_consumption: str ('none', 'light', 'moderate', 'heavy')
            - family_history_diabetes: int (0 o 1)
            - fasting_glucose: float (mg/dL)
            - systolic_bp: float (mmHg)
            - triglycerides: float (mg/dL)

        Returns:
        --------
        dict
            Diccionario con la predicción y probabilidades
        """
        try:
            # Preprocesar datos
            X = self._preprocess_input(data)

            # Verificar que no hay valores nulos
            if X.isnull().any().any():
                raise ValueError(
                    "Hay valores nulos después del preprocesamiento")

            # Obtener predicciones de cada modelo
            rf_pred = self.rf_model.predict_proba(X)[:, 1]
            lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
            xgb_pred = self.xgb_model.predict_proba(X)[:, 1]

            # Ajustar pesos si HbA1c fue imputado
            if 'hba1c' not in data or pd.isna(data.get('hba1c')):
                # Ajustar pesos para dar más importancia a modelos que dependen menos de HbA1c
                # RF y LGB tienen más peso
                adjusted_weights = np.array([0.4, 0.4, 0.2])
                print("DEBUG - Usando pesos ajustados para predicción sin HbA1c")
            else:
                adjusted_weights = self.weights

            # Calcular predicción del ensemble
            ensemble_prob = (
                adjusted_weights[0] * rf_pred +
                adjusted_weights[1] * lgb_pred +
                adjusted_weights[2] * xgb_pred
            )[0]

            # Obtener y validar valores clínicos críticos
            try:
                hba1c = float(data['hba1c'])
                fasting_glucose = float(data['fasting_glucose'])
                print(
                    f"DEBUG - Valores clínicos: HbA1c={hba1c}, Glucosa={fasting_glucose}")
            except (ValueError, TypeError) as e:
                print(f"ERROR - Conversión de valores clínicos: {str(e)}")
                raise ValueError("Error en la conversión de valores clínicos")

            # PRIMERO: Aplicar criterios clínicos de la ADA
            print(
                f"DEBUG - Evaluando criterios de diabetes: HbA1c >= 6.5 ({hba1c >= 6.5}) o Glucosa >= 126 ({fasting_glucose >= 126})")

            # Forzar la clasificación basada en criterios clínicos
            if hba1c >= 6.5 or fasting_glucose >= 126:
                print(
                    "DEBUG - CLASIFICACIÓN: Diabetes confirmada por criterios clínicos")
                prediction = "Diabetes"
                risk_level = "Alto"
                ensemble_prob = 0.85
                recommendations = [
                    "Consulta médica inmediata",
                    "Control de glucemia frecuente",
                    "Evaluación de complicaciones",
                    "Plan de alimentación específico"
                ]

            # SEGUNDO: Evaluar prediabetes si no es diabetes
            elif (5.7 <= hba1c < 6.5) or (100 <= fasting_glucose < 126):
                print("DEBUG - CLASIFICACIÓN: Pre-diabetes por criterios clínicos")
                prediction = "Pre-diabetes"

                # Calcular score de riesgo
                risk_score = 0
                risk_factors = 0

                # Factores primarios
                if float(data['bmi']) >= 30:
                    risk_score += 0.05
                    risk_factors += 1
                elif float(data['bmi']) >= 25:
                    risk_score += 0.025
                    risk_factors += 0.5
                if data['family_history_diabetes'] == 1:
                    risk_score += 0.05
                    risk_factors += 1

                # Factores secundarios
                if data['smoker'] == 1:
                    risk_score += 0.025
                    risk_factors += 0.5
                if float(data['triglycerides']) >= 200:
                    risk_score += 0.025
                    risk_factors += 0.5
                elif float(data['triglycerides']) >= 150:
                    risk_score += 0.015
                    risk_factors += 0.25
                if float(data['age']) >= 45:
                    risk_score += 0.025
                    risk_factors += 0.5

                # Ajustar nivel de riesgo basado en el score
                if risk_score >= 0.1:
                    risk_level = "Alto"
                elif risk_score >= 0.05:
                    risk_level = "Moderado"
                else:
                    risk_level = "Bajo"

                # Generar recomendaciones basadas en el nivel de riesgo
                recommendations = [
                    "Control de peso",
                    "Actividad física regular",
                    "Dieta balanceada"
                ]
                if risk_level == "Alto":
                    recommendations.extend([
                        "Control de glucemia frecuente",
                        "Consulta médica en 3 meses"
                    ])
                elif risk_level == "Moderado":
                    recommendations.extend([
                        "Control de glucemia cada 6 meses",
                        "Consulta médica en 6 meses"
                    ])

            # TERCERO: Usar el modelo de machine learning para casos no concluyentes
            else:
                print("DEBUG - CLASIFICACIÓN: Usando modelo de machine learning")
                if ensemble_prob >= self.threshold:
                    prediction = "Diabetes"
                    risk_level = "Alto"
                    recommendations = [
                        "Control de peso",
                        "Actividad física regular",
                        "Dieta balanceada",
                        "Control de glucemia frecuente",
                        "Consulta médica en 3 meses"
                    ]
                else:
                    prediction = "No Diabetes"
                    risk_level = "Bajo"
                    recommendations = [
                        "Mantener estilo de vida saludable",
                        "Control de peso",
                        "Actividad física regular"
                    ]

            return {
                "prediction": prediction,
                "probability": float(ensemble_prob),
                "risk_level": risk_level,
                "recommendations": recommendations,
                "hba1c_imputed": 'hba1c' not in data or pd.isna(data.get('hba1c')),
                "model_probabilities": {
                    "random_forest": float(rf_pred[0]),
                    "lightgbm": float(lgb_pred[0]),
                    "xgboost": float(xgb_pred[0])
                }
            }

        except Exception as e:
            print(f"Error en la predicción: {str(e)}")
            raise


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de datos de entrada
    sample_data = {
        'age': 45,
        'gender': 'male',
        'height': 175,
        'weight': 85,
        'bmi': 27.8,
        'hba1c': 6.0,
        'physical_activity': 'light',
        'smoker': 1,
        'alcohol_consumption': 'moderate',
        'family_history_diabetes': 0,
        'fasting_glucose': 110,
        'systolic_bp': 130,
        'triglycerides': 160
    }

    # Crear instancia del predictor
    predictor = DiabetesPredictor()

    # Realizar predicción
    try:
        result = predictor.predict(sample_data)
        print("\nResultado de la predicción:")
        print(f"Predicción: {result['prediction']}")
        print(f"Probabilidad: {result['probability']:.2%}")
        print(f"Nivel de Riesgo: {result['risk_level']}")
        print("\nRecomendaciones:")
        for rec in result['recommendations']:
            print(f"- {rec}")
        print("\nProbabilidades por modelo:")
        for model, prob in result['model_probabilities'].items():
            print(f"{model}: {prob:.2%}")
    except Exception as e:
        print(f"Error: {str(e)}")
