import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, roc_curve, auc
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
from app.utils.obesity.obesity_model import create_target_variables
import matplotlib.pyplot as plt
from scipy import stats


def prepare_data(df):
    """
    Prepara los datos para el modelo usando discretización y encoding
    Implementa una versión más conservadora para evitar sobreajuste
    """
    # Crear solo variables objetivo (sin features derivadas para evitar data leakage)
    df = create_target_variables(df)

    # Codificar variables categóricas
    le = LabelEncoder()
    categorical_features = [
        'gender', 'physical_activity', 'alcohol_consumption']
    for feature in categorical_features:
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])

    # Función auxiliar para discretización segura con menos bins
    def safe_qcut(series, q, name):
        try:
            return pd.qcut(series, q=q, labels=False, duplicates='drop')
        except ValueError as e:
            print(
                f"Advertencia: Problema al discretizar {name}. Usando alternativa...")
            try:
                return pd.qcut(series, q=3, labels=False, duplicates='drop')
            except ValueError:
                return pd.cut(series, bins=3, labels=False)

    # Discretizar variables numéricas con menos categorías
    print("Discretizando variables numéricas...")
    df['bmi_cat'] = safe_qcut(df['bmi'], 3, 'BMI')
    df['age_cat'] = safe_qcut(df['age'], 3, 'Age')
    df['glucose_cat'] = safe_qcut(df['fasting_glucose'], 3, 'Glucose')
    df['bp_cat'] = safe_qcut(df['systolic_bp'], 3, 'Blood Pressure')

    # Features base (sin características derivadas complejas)
    features = [
        'bmi_cat', 'age_cat', 'glucose_cat', 'bp_cat',
        'gender_encoded', 'physical_activity_encoded',
        'alcohol_consumption_encoded', 'smoker',
        'family_history_obesity'
    ]

    # Verificar valores nulos
    null_features = df[features].columns[df[features].isnull().any()].tolist()
    if null_features:
        print(f"Advertencia: Valores nulos encontrados en: {null_features}")
        for feature in null_features:
            if df[feature].dtype in ['int64', 'float64']:
                df[feature] = df[feature].fillna(df[feature].median())
            else:
                df[feature] = df[feature].fillna(df[feature].mode()[0])

    # Escalar features numéricas
    scaler = StandardScaler()
    numeric_features = ['bmi_cat', 'age_cat', 'glucose_cat', 'bp_cat']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    print("Preparación de datos completada.")
    return df, features


def plot_roc_curves(y_test, y_probs_dict, ensemble_probs):
    plt.figure(figsize=(10, 8))
    colors = {'random_forest': 'blue', 'lightgbm': 'green',
              'xgboost': 'red', 'ensemble': 'purple'}

    # Calcular y graficar ROC para cada modelo
    for model_name, y_prob in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        p_value = calculate_roc_p_value(y_test, y_prob)
        plt.plot(fpr, tpr, color=colors[model_name],
                 label=f'{model_name} (AUC = {roc_auc:.3f}, p = {p_value:.3e})')

    # Graficar ROC del ensemble
    fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
    ensemble_auc = auc(fpr, tpr)
    ensemble_p = calculate_roc_p_value(y_test, ensemble_probs)
    plt.plot(fpr, tpr, color=colors['ensemble'],
             label=f'Ensemble (AUC = {ensemble_auc:.3f}, p = {ensemble_p:.3e})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Comparación de Modelos')
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))

    # Crear directorios para los modelos si no existen
    models_dir = os.path.join('app', 'models', 'obesity')
    os.makedirs(models_dir, exist_ok=True)

    # Guardar la curva ROC
    plt.savefig(os.path.join(models_dir, 'roc_curves.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def calculate_roc_p_value(y_true, y_pred):
    """
    Calcula el p-value para la curva ROC usando el método de bootstrap
    """
    # Convertir a numpy arrays si son pandas Series
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_bootstraps = 1000
    roc_auc_scores = []
    n_samples = len(y_true)

    # Bootstrap para calcular distribución del AUC
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]

        if len(np.unique(y_true_bootstrap)) < 2:
            continue

        try:
            roc_auc = roc_auc_score(y_true_bootstrap, y_pred_bootstrap)
            roc_auc_scores.append(roc_auc)
        except ValueError:
            continue

    if not roc_auc_scores:
        return 1.0  # Si no se pueden calcular scores, retornar 1.0

    # Calcular p-value
    mean_auc = np.mean(roc_auc_scores)
    p_value = (np.sum(np.array(roc_auc_scores) <= 0.5) + 1) / \
        (n_bootstraps + 1)
    return p_value


def train_ensemble(X_train, X_test, y_train, y_test, feature_names):
    """
    Entrena un modelo de ensamble usando Random Forest, LGBM y XGBoost
    con parámetros muy conservadores para evitar sobreajuste

    Parameters:
    -----------
    X_train : array-like
        Features de entrenamiento
    X_test : array-like
        Features de prueba
    y_train : array-like
        Etiquetas de entrenamiento
    y_test : array-like
        Etiquetas de prueba
    feature_names : list
        Lista de nombres de las features utilizadas
    """
    os.makedirs('models', exist_ok=True)

    # SMOTE con configuración más conservadora
    # Submuestreo más conservador
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Parámetros ultra conservadores para Random Forest
    rf_params = {
        'n_estimators': 50,  # Menos árboles
        'max_depth': 2,      # Profundidad muy limitada
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',  # Menos features por árbol
        'class_weight': 'balanced',
        'random_state': 42
    }

    # Parámetros ultra conservadores para LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 4,     # Muy pocos leaves
        'learning_rate': 0.005,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 5,
        'min_child_samples': 100,  # Más muestras por nodo
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'max_depth': 2,      # Profundidad muy limitada
        'verbose': -1
    }

    # Parámetros ultra conservadores para XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 2,      # Profundidad muy limitada
        'learning_rate': 0.005,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'min_child_weight': 10,
        'gamma': 2.0,        # Mayor regularización
        'alpha': 2.0,
        'lambda': 2.0,
        'scale_pos_weight': 1.0
    }

    # Validación cruzada más estricta
    n_splits = 10  # Más folds para mejor validación
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Random Forest
    rf_model = RandomForestClassifier(**rf_params)
    rf_scores = cross_val_score(rf_model, X_train, y_train,  # Sin usar datos balanceados para CV
                                cv=skf, scoring='roc_auc')
    rf_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"Random Forest CV AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_scores = cross_val_score(lgb_model, X_train, y_train,  # Sin usar datos balanceados para CV
                                 cv=skf, scoring='roc_auc')
    lgb_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"LightGBM CV AUC: {lgb_scores.mean():.4f} (+/- {lgb_scores.std() * 2:.4f})")

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_scores = cross_val_score(xgb_model, X_train, y_train,  # Sin usar datos balanceados para CV
                                 cv=skf, scoring='roc_auc')
    xgb_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"XGBoost CV AUC: {xgb_scores.mean():.4f} (+/- {xgb_scores.std() * 2:.4f})")

    # Predicciones en datos no balanceados
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

    # Pesos basados en la varianza de los scores
    weights = np.array(
        [1/rf_scores.std(), 1/lgb_scores.std(), 1/xgb_scores.std()])
    weights = weights / weights.sum()

    ensemble_pred = (weights[0] * rf_pred +
                     weights[1] * lgb_pred +
                     weights[2] * xgb_pred)

    # Encontrar mejor umbral
    precision, recall, thresholds = precision_recall_curve(
        y_test, ensemble_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])]

    # Crear directorios para los modelos si no existen
    models_dir = os.path.join('app', 'models', 'obesity')
    os.makedirs(models_dir, exist_ok=True)

    # Guardar modelos
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))
    joblib.dump(lgb_model, os.path.join(models_dir, 'lgb_model.pkl'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))

    # Guardar también los pesos y el umbral
    model_config = {
        'weights': weights.tolist(),
        'threshold': best_threshold
    }
    with open(os.path.join(models_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f)

    # Evaluación final
    print(f"\nMejor umbral: {best_threshold:.4f}")
    print("\nInforme de clasificación:")
    print(classification_report(
        y_test, (ensemble_pred >= best_threshold).astype(int)))
    print(f"ROC AUC Score: {roc_auc_score(y_test, ensemble_pred):.4f}")

    # Imprimir importancia de features
    print("\nImportancia de features por modelo:")
    for model_name, model in [('Random Forest', rf_model), ('LightGBM', lgb_model), ('XGBoost', xgb_model)]:
        print(f"\n{model_name.upper()}:")
        importances = dict(zip(feature_names, model.feature_importances_))
        sorted_features = sorted(
            importances.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")

    # Generar y guardar curva ROC
    plot_roc_curves(y_test, {'random_forest': rf_pred,
                    'lightgbm': lgb_pred, 'xgboost': xgb_pred}, ensemble_pred)

    return {
        'rf_model': rf_model,
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'weights': weights,
        'threshold': best_threshold
    }


def main():
    # Cargar dataset
    print("Cargando dataset...")
    df = pd.read_csv('app/datasets/final_dataset.csv')

    # Preparar datos
    df, features = prepare_data(df)

    # División de datos
    X = df[features]
    y = df['obesity']

    # División estratificada con mayor porcentaje de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Entrenar modelo
    print("\nEntrenando modelos...")
    model = train_ensemble(X_train, X_test, y_train, y_test, features)

    print("\nModelos guardados en el directorio 'models/'")
    print("Curva ROC guardada como 'models/roc_curves.png'")


if __name__ == "__main__":
    main()
