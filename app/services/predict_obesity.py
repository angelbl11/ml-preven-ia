import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import json

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


class ObesityPredictor:
    def __init__(self):
        # Cargar modelos usando rutas absolutas
        models_dir = os.path.join(BASE_DIR, 'app', 'models', 'obesity')
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
                "Warning: No se pudo cargar la configuración del modelo, usando valores actualizados")
            # Ajustando los pesos para dar más importancia al BMI
            self.weights = np.array([0.4, 0.3, 0.3])  # RF, LGB, XGB
            self.threshold = 0.35  # Bajando el threshold para ser más sensible

        # Inicializar transformadores
        self.le = LabelEncoder()
        self.scaler = StandardScaler()

        # Definir rangos para discretización
        self.bmi_bins = [0, 18.5, 25, 30, 35, 40, float('inf')]
        self.age_bins = [0, 30, 50, float('inf')]
        self.glucose_bins = [0, 100, 126, float('inf')]
        self.bp_bins = [0, 120, 140, float('inf')]

    def _safe_transform(self, value, feature_name, bins):
        """Discretiza un valor de forma segura"""
        try:
            # Asegurarse de que el valor es numérico
            value = float(value)
            # Usar pd.cut para discretización
            cat = pd.cut([value], bins=bins, labels=False)[0]
            # Manejar valores nulos
            return 1 if pd.isna(cat) else int(cat)
        except (ValueError, TypeError):
            print(
                f"Warning: Error al transformar {feature_name}, usando valor por defecto")
            return 1  # Valor por defecto medio

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
        df['bmi_cat'] = self._safe_transform(data['bmi'], 'bmi', self.bmi_bins)
        df['age_cat'] = self._safe_transform(data['age'], 'age', self.age_bins)
        df['glucose_cat'] = self._safe_transform(
            data['fasting_glucose'], 'glucose', self.glucose_bins)
        df['bp_cat'] = self._safe_transform(
            data['systolic_bp'], 'bp', self.bp_bins)

        # Codificar variables categóricas
        categorical_mapping = {
            'gender': {'male': 0, 'female': 1},
            'physical_activity': {'none': 0, 'light': 1, 'moderate': 2, 'frequent': 3},
            'alcohol_consumption': {'none': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
        }

        # Aplicar mapeo categórico
        for feature, mapping in categorical_mapping.items():
            df[f'{feature}_encoded'] = df[feature].map(mapping).fillna(0)

        # Asegurar que smoker y family_history son numéricos
        df['smoker'] = pd.to_numeric(df['smoker'], errors='coerce').fillna(0)
        df['family_history_obesity'] = pd.to_numeric(
            df['family_history_obesity'], errors='coerce').fillna(0)

        # Seleccionar y ordenar features
        features = [
            'bmi_cat', 'age_cat', 'glucose_cat', 'bp_cat',
            'gender_encoded', 'physical_activity_encoded',
            'alcohol_consumption_encoded', 'smoker',
            'family_history_obesity'
        ]

        # Asegurar que no hay valores nulos
        for feature in features:
            if df[feature].isnull().any():
                df[feature] = df[feature].fillna(0)

        # Escalar features numéricas
        numeric_features = ['bmi_cat', 'age_cat', 'glucose_cat', 'bp_cat']
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])

        return df[features]

    def _get_bmi_category(self, bmi):
        """Determina la categoría de IMC"""
        if bmi < 18.5:
            return "Bajo peso"
        elif bmi < 25:
            return "Peso normal"
        elif bmi < 30:
            return "Sobrepeso"
        elif bmi < 35:
            return "Obesidad Tipo 1"
        elif bmi < 40:
            return "Obesidad Tipo 2"
        else:
            return "Obesidad Tipo 3"

    def _calculate_bmi_probability(self, bmi):
        """Calcula una probabilidad basada en el BMI"""
        if bmi >= 40:  # Obesidad Tipo 3
            return 0.99  # Aumentado a casi certeza
        elif bmi >= 35:  # Obesidad Tipo 2
            return 0.95  # Aumentado
        elif bmi >= 30:  # Obesidad Tipo 1
            return 0.90  # Aumentado
        elif bmi >= 25:  # Sobrepeso
            return 0.70  # Aumentado
        else:  # Peso normal o bajo peso
            return 0.15

    def _calculate_risk_factor_probability(self, data):
        """Calcula una probabilidad adicional basada en factores de riesgo"""
        base_prob = 0.50  # Comenzamos con una base más alta

        # Factores de riesgo principales
        if data['physical_activity'] == 'none':
            base_prob += 0.20
        elif data['physical_activity'] == 'light':
            base_prob += 0.15

        if data['smoker']:
            base_prob += 0.15

        if data['family_history_obesity']:
            base_prob += 0.20

        # Consumo de alcohol
        if data['alcohol_consumption'] == 'heavy':
            base_prob += 0.20
        elif data['alcohol_consumption'] == 'moderate':
            base_prob += 0.15

        # Factores metabólicos
        if data['fasting_glucose'] > 126:  # Diabetes
            base_prob += 0.20
        elif data['fasting_glucose'] > 100:  # Prediabetes
            base_prob += 0.15

        if data['systolic_bp'] > 140:  # Hipertensión
            base_prob += 0.20
        elif data['systolic_bp'] > 130:  # Prehipertensión
            base_prob += 0.15

        # Normalizar la probabilidad para que no exceda 1
        return min(base_prob, 0.99)

    def predict(self, data: dict) -> dict:
        """
        Realiza una predicción de obesidad usando el ensemble de modelos.

        Parameters:
        -----------
        data : dict
            Diccionario con los datos del paciente:
            - age: int
                Edad del paciente en años
            - gender: str
                Género del paciente ('male' o 'female')
            - height: float
                Altura del paciente en centímetros
            - weight: float
                Peso del paciente en kilogramos
            - bmi: float
                Índice de masa corporal
            - physical_activity: str
                Nivel de actividad física ('none', 'light', 'moderate', 'frequent')
            - smoker: int
                Indica si el paciente fuma (0 o 1)
            - alcohol_consumption: str
                Nivel de consumo de alcohol ('none', 'light', 'moderate', 'heavy')
            - family_history_obesity: int
                Indica si hay historia familiar de obesidad (0 o 1)
            - fasting_glucose: float
                Nivel de glucosa en ayunas en mg/dL
            - systolic_bp: float
                Presión arterial sistólica en mmHg

        Returns:
        --------
        dict
            Diccionario con la predicción y probabilidades:
            - prediction: str
                Categoría de IMC ('Bajo peso', 'Peso normal', 'Sobrepeso', 'Obesidad Tipo 1', 'Obesidad Tipo 2', 'Obesidad Tipo 3')
            - probability: float
                Probabilidad de obesidad (0-1)
            - risk_level: str
                Nivel de riesgo ('Muy Alto', 'Alto', 'Moderado', 'Bajo')
            - recommendations: list[str]
                Lista de recomendaciones personalizadas
            - model_probabilities: dict
                Probabilidades de cada modelo individual:
                - random_forest: float
                - lightgbm: float
                - xgboost: float
                - bmi_based: float
                - risk_factors: float

        Example:
        --------
        >>> predictor = ObesityPredictor()
        >>> sample_data = {
        ...     'age': 45,
        ...     'gender': 'male',
        ...     'height': 175,
        ...     'weight': 85,
        ...     'bmi': 27.8,
        ...     'physical_activity': 'light',
        ...     'smoker': 1,
        ...     'alcohol_consumption': 'moderate',
        ...     'family_history_obesity': 0,
        ...     'fasting_glucose': 95,
        ...     'systolic_bp': 130
        ... }
        >>> result = predictor.predict(sample_data)
        >>> print(result['prediction'])
        'Sobrepeso'
        >>> print(result['probability'])
        0.70
        >>> print(result['risk_level'])
        'Moderado'
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

            # Calcular predicción del ensemble
            ensemble_prob = (
                self.weights[0] * rf_pred +
                self.weights[1] * lgb_pred +
                self.weights[2] * xgb_pred
            )

            # Calcular las diferentes probabilidades
            bmi_prob = self._calculate_bmi_probability(data['bmi'])
            risk_prob = self._calculate_risk_factor_probability(data)

            # Ajustamos los pesos: 70% BMI, 25% factores de riesgo, 5% ensemble
            final_prob = (0.70 * bmi_prob) + \
                (0.25 * risk_prob) + (0.05 * ensemble_prob)

            # Para casos de obesidad tipo 2 y 3, aseguramos un mínimo de probabilidad
            if data['bmi'] >= 35:
                final_prob = max(final_prob, 0.90)
            elif data['bmi'] >= 30:
                final_prob = max(final_prob, 0.85)

            # Determinar la categoría de IMC
            bmi_category = self._get_bmi_category(data['bmi'])

            # Ajustar factores de riesgo adicionales
            risk_factors = sum([
                data['smoker'],
                data['family_history_obesity'],
                data['physical_activity'] == 'none',
                data['alcohol_consumption'] in ['moderate', 'heavy'],
                data['fasting_glucose'] > 100,
                data['systolic_bp'] > 130
            ])

            # Determinar nivel de riesgo y recomendaciones
            if data['bmi'] >= 35:
                risk_level = "Muy Alto"
                recommendations = [
                    "Consulta médica inmediata con especialista en obesidad",
                    "Evaluación para posible intervención quirúrgica",
                    "Plan de alimentación especializado",
                    "Programa de ejercicio supervisado por profesional",
                    "Evaluación psicológica para manejo de conducta alimentaria",
                    "Control estricto de comorbilidades"
                ]
            elif data['bmi'] >= 30:
                risk_level = "Alto"
                recommendations = [
                    "Consulta médica con especialista en nutrición",
                    "Plan de alimentación personalizado",
                    "Programa de ejercicio supervisado",
                    "Evaluación de comorbilidades asociadas",
                    "Seguimiento mensual con equipo médico"
                ]
            elif data['bmi'] >= 25:
                risk_level = "Moderado"
                recommendations = [
                    "Consulta con nutricionista",
                    "Incrementar actividad física gradualmente",
                    "Modificación de hábitos alimenticios",
                    "Control regular de peso y medidas",
                    "Evaluación trimestral de progreso"
                ]
            else:
                risk_level = "Bajo"
                recommendations = [
                    "Mantener peso saludable",
                    "Actividad física regular",
                    "Alimentación balanceada",
                    "Control periódico de IMC",
                    "Prevención de factores de riesgo"
                ]

            # Ajustar recomendaciones según factores de riesgo adicionales
            if risk_factors >= 3:
                recommendations.append(
                    "Evaluación integral de factores de riesgo cardiovascular")
            if data['fasting_glucose'] > 100:
                recommendations.append(
                    "Control periódico de glucosa en ayunas")
            if data['systolic_bp'] > 130:
                recommendations.append("Monitoreo regular de presión arterial")

            return {
                'prediction': bmi_category,
                'probability': float(final_prob),
                'risk_level': risk_level,
                'recommendations': recommendations,
                'model_probabilities': {
                    'random_forest': float(rf_pred),
                    'lightgbm': float(lgb_pred),
                    'xgboost': float(xgb_pred),
                    'bmi_based': float(bmi_prob),
                    'risk_factors': float(risk_prob)
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
        'physical_activity': 'light',
        'smoker': 1,
        'alcohol_consumption': 'moderate',
        'family_history_obesity': 0,
        'fasting_glucose': 95,
        'systolic_bp': 130
    }

    # Crear instancia del predictor
    predictor = ObesityPredictor()

    # Realizar predicción
    try:
        result = predictor.predict(sample_data)
        print("\nResultado de la predicción:")
        print(
            f"Predicción: {result['prediction']}")
        print(f"Probabilidad: {result['probability']:.2%}")
        print(f"Nivel de Riesgo: {result['risk_level']}")
        print("\nRecomendaciones:")
        for recommendation in result['recommendations']:
            print(f"- {recommendation}")
        print("\nProbabilidades por modelo:")
        for model, prob in result['model_probabilities'].items():
            print(f"{model}: {prob:.2%}")
    except Exception as e:
        print(f"Error: {str(e)}")
