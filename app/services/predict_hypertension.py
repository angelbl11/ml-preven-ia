import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
import json

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


class HypertensionPredictor:
    def __init__(self):
        # Cargar modelos usando rutas absolutas
        models_dir = os.path.join(BASE_DIR, 'app', 'models', 'hypertension')

        # Crear directorio de modelos si no existe
        os.makedirs(models_dir, exist_ok=True)

        try:
            self.rf_model = joblib.load(
                os.path.join(models_dir, 'rf_model.pkl'))
            self.lgb_model = joblib.load(
                os.path.join(models_dir, 'lgb_model.pkl'))
            self.xgb_model = joblib.load(
                os.path.join(models_dir, 'xgb_model.pkl'))
        except (FileNotFoundError, EOFError, ImportError) as e:
            print(
                f"Warning: No se pudieron cargar los modelos. Error: {str(e)}")
            print(
                "Por favor, asegúrese de entrenar los modelos primero usando 'make train-hypertension'")
            # Crear modelos dummy para desarrollo
            self.rf_model = RandomForestClassifier(n_estimators=10)
            self.lgb_model = lgb.LGBMClassifier(n_estimators=10)
            self.xgb_model = xgb.XGBClassifier(n_estimators=10)

            # Entrenar con datos dummy para que funcionen las predicciones
            X_dummy = np.random.rand(100, 7)
            y_dummy = np.random.randint(0, 2, 100)

            self.rf_model.fit(X_dummy, y_dummy)
            self.lgb_model.fit(X_dummy, y_dummy)
            self.xgb_model.fit(X_dummy, y_dummy)

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
        # Normal/Pre-HTA, HTA 1, HTA 2
        self.systolic_bins = [0, 130, 160, float('inf')]
        # Normal/Pre-HTA, HTA 1, HTA 2
        self.diastolic_bins = [0, 85, 100, float('inf')]
        self.age_bins = [0, 60, float('inf')]  # No mayor, Mayor
        self.bmi_bins = [0, 30, float('inf')]  # No obeso, Obeso

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

        # Discretizar variables numéricas de forma segura
        df['systolic_bp_cat'] = self._safe_transform(
            data['systolic_bp'], 'systolic_bp', self.systolic_bins)
        df['diastolic_bp_cat'] = self._safe_transform(
            data['diastolic_bp'], 'diastolic_bp', self.diastolic_bins)
        df['age_cat'] = self._safe_transform(data['age'], 'age', self.age_bins)
        df['bmi_cat'] = self._safe_transform(data['bmi'], 'bmi', self.bmi_bins)

        # Codificar variables categóricas
        categorical_mapping = {
            'gender': {'male': 0, 'female': 1}
        }

        # Aplicar mapeo categórico
        for feature, mapping in categorical_mapping.items():
            df[f'{feature}_encoded'] = df[feature].map(mapping).fillna(0)

        # Calcular características combinadas
        df['bp_combined'] = df['systolic_bp_cat'].astype(
            int) + df['diastolic_bp_cat'].astype(int)

        # Calcular índice de riesgo cardiovascular
        risk_factors = ['smoker', 'family_history_hypertension']
        df['cv_risk'] = sum(pd.to_numeric(df[factor], errors='coerce').fillna(0)
                            for factor in risk_factors if factor in data)

        # Seleccionar y ordenar features
        features = [
            'systolic_bp_cat',
            'diastolic_bp_cat',
            'bp_combined',
            'age_cat',
            'bmi_cat',
            'cv_risk',
            'gender_encoded'
        ]

        # Asegurar que no hay valores nulos
        for feature in features:
            if df[feature].isnull().any():
                df[feature] = df[feature].fillna(0)

        # Escalar features numéricas
        numeric_features = ['systolic_bp_cat', 'diastolic_bp_cat',
                            'bp_combined', 'age_cat', 'bmi_cat', 'cv_risk']
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])

        return df[features]

    def predict(self, data):
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

            # Calcular predicción del ensemble
            ensemble_prob = (
                self.weights[0] * rf_pred +
                self.weights[1] * lgb_pred +
                self.weights[2] * xgb_pred
            )[0]

            # Obtener valores de presión arterial
            systolic_bp = float(data['systolic_bp'])
            diastolic_bp = float(data['diastolic_bp'])

            # Calcular factores de riesgo
            risk_factors = 0
            if float(data['bmi']) >= 30:  # Obesidad
                risk_factors += 1
            if data['smoker'] == 1:  # Fumador
                risk_factors += 1
            if data['alcohol_consumption'] in ['moderate', 'heavy']:  # Alcohol
                risk_factors += 1
            if data['family_history_hypertension'] == 1:  # Historia familiar
                risk_factors += 1
            if data['physical_activity'] == 'none':  # Sedentarismo
                risk_factors += 1
            if float(data['fasting_glucose']) >= 126:  # Diabetes
                risk_factors += 1

            # Clasificación según criterios JNC 7 y actualizaciones recientes
            if systolic_bp >= 180 or diastolic_bp >= 120:
                prediction = "Crisis Hipertensiva"
                risk_level = "Muy Alto"
                ensemble_prob = 0.95
                recommendations = [
                    "EMERGENCIA MÉDICA - Buscar atención inmediata",
                    "Monitoreo continuo de presión arterial",
                    "Evaluación urgente de daño a órganos",
                    "Tratamiento inmediato para reducir PA"
                ]
            elif systolic_bp >= 160 or diastolic_bp >= 100:
                prediction = "Hipertensión Grado 3"
                risk_level = "Alto"
                ensemble_prob = 0.85
                recommendations = [
                    "Consulta médica inmediata",
                    "Evaluación cardiovascular completa",
                    "Inicio/ajuste de medicación",
                    "Cambios intensivos en estilo de vida"
                ]
            elif systolic_bp >= 150 or diastolic_bp >= 95:
                prediction = "Hipertensión Grado 2"
                risk_level = "Medio-Alto"
                ensemble_prob = 0.75
                recommendations = [
                    "Consulta médica en la próxima semana",
                    "Monitoreo regular de presión arterial",
                    "Evaluación de factores de riesgo",
                    "Modificación de estilo de vida"
                ]
            elif systolic_bp >= 140 or diastolic_bp >= 90:
                prediction = "Hipertensión Grado 1"
                risk_level = "Medio"
                ensemble_prob = 0.65
                recommendations = [
                    "Consulta médica en las próximas 2-3 semanas",
                    "Monitoreo de presión arterial",
                    "Plan de reducción de sodio",
                    "Incremento de actividad física"
                ]
            elif systolic_bp >= 130 or diastolic_bp >= 85:
                # Ajustar probabilidad basada en factores de riesgo
                base_prob = 0.45
                risk_prob = min(base_prob + (risk_factors * 0.05), 0.60)

                if risk_factors >= 3:
                    prediction = "Hipertensión Grado 1"
                    risk_level = "Medio"
                    ensemble_prob = risk_prob
                    recommendations = [
                        "Consulta médica preventiva",
                        "Monitoreo regular de presión arterial",
                        "Evaluación de factores de riesgo",
                        "Modificación de estilo de vida"
                    ]
                else:
                    prediction = "Pre-hipertensión"
                    risk_level = "Bajo-Medio"
                    ensemble_prob = risk_prob
                    recommendations = [
                        "Control de presión arterial periódico",
                        "Evaluación de factores de riesgo",
                        "Modificación gradual del estilo de vida",
                        "Dieta baja en sodio"
                    ]
            else:
                prediction = "No Hipertensión"
                risk_level = "Bajo"
                # Ajustar probabilidad basada en factores de riesgo, pero manteniéndola baja
                ensemble_prob = min(0.2 + (risk_factors * 0.03), 0.35)
                recommendations = [
                    "Control anual de presión arterial",
                    "Mantener estilo de vida saludable",
                    "Dieta balanceada",
                    "Actividad física regular"
                ]

            # Agregar recomendaciones específicas basadas en factores de riesgo
            specific_recommendations = []

            if float(data['bmi']) >= 30:
                specific_recommendations.append(
                    "Plan de reducción de peso supervisado médicamente")
            elif float(data['bmi']) >= 25:
                specific_recommendations.append(
                    "Control y reducción gradual del peso")

            if data['smoker'] == 1:
                specific_recommendations.append(
                    "Cesación inmediata del tabaquismo")

            if data['alcohol_consumption'] in ['moderate', 'heavy']:
                specific_recommendations.append(
                    "Reducción del consumo de alcohol")

            if data['physical_activity'] == 'none':
                specific_recommendations.append(
                    "Inicio gradual de actividad física")

            # Combinar recomendaciones según la severidad
            if prediction in ["Crisis Hipertensiva", "Hipertensión Grado 3"]:
                # Mantener las recomendaciones críticas
                recommendations = recommendations[:3] + [
                    specific_recommendations[0]] if specific_recommendations else recommendations
            else:
                # Incluir recomendaciones específicas
                if specific_recommendations:
                    recommendations = recommendations[:2] + \
                        specific_recommendations[:2]

            return {
                'prediction': prediction,
                'probability': float(ensemble_prob),
                'risk_level': risk_level,
                'recommendations': recommendations,
                'model_probabilities': {
                    'random_forest': float(rf_pred[0]),
                    'lightgbm': float(lgb_pred[0]),
                    'xgboost': float(xgb_pred[0])
                }
            }

        except Exception as e:
            print(f"Error en la predicción: {str(e)}")
            raise ValueError(f"Error al procesar la predicción: {str(e)}")


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de datos de entrada
    sample_data = {
        'age': 45,
        'gender': 'male',
        'height': 175,
        'weight': 85,
        'bmi': 27.8,
        'systolic_bp': 145,
        'diastolic_bp': 95,
        'physical_activity': 'light',
        'smoker': 1,
        'alcohol_consumption': 'moderate',
        'family_history_hypertension': 1
    }

    # Crear instancia del predictor
    predictor = HypertensionPredictor()

    # Realizar predicción
    try:
        result = predictor.predict(sample_data)
        print("\nResultado de la predicción:")
        print(
            f"Predicción: {'Hipertensión' if result['prediction'] == 1 else 'No Hipertensión'}")
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
