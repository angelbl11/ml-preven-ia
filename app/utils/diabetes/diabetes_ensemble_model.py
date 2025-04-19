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
from app.utils.diabetes.diabetes_model import create_target_variables
import matplotlib.pyplot as plt
from scipy import stats


def prepare_data(df):
    """
    Prepara los datos para el modelo usando discretización y encoding
    Implementa una versión más conservadora para evitar sobreajuste
    """
    # Filtrar pacientes entre 18 y 65 años
    df = df[(df['age'] >= 18) & (df['age'] <= 65)].copy()

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
    df['hba1c_cat'] = safe_qcut(df['hba1c'], 3, 'HbA1c')
    df['glucose_cat'] = safe_qcut(df['fasting_glucose'], 3, 'Glucose')
    df['bmi_cat'] = safe_qcut(df['bmi'], 3, 'BMI')
    df['age_cat'] = safe_qcut(df['age'], 3, 'Age')
    df['triglycerides_cat'] = safe_qcut(
        df['triglycerides'], 3, 'Triglycerides')

    # Features base (sin características derivadas complejas)
    features = [
        'hba1c_cat', 'glucose_cat', 'bmi_cat', 'age_cat', 'triglycerides_cat',
        'gender_encoded', 'physical_activity_encoded',
        'alcohol_consumption_encoded', 'smoker',
        'family_history_diabetes'
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
    numeric_features = ['hba1c_cat', 'glucose_cat',
                        'bmi_cat', 'age_cat', 'triglycerides_cat']
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
        # Calcular precisión
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = np.mean(precision)
        plt.plot(fpr, tpr, color=colors[model_name],
                 label=f'{model_name} (AUC = {roc_auc:.3f}, p = {p_value:.3e}, Precision = {avg_precision:.3f})')

    # Graficar ROC del ensemble
    fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
    ensemble_auc = auc(fpr, tpr)
    ensemble_p = calculate_roc_p_value(y_test, ensemble_probs)
    # Calcular precisión del ensemble
    precision, recall, _ = precision_recall_curve(y_test, ensemble_probs)
    ensemble_avg_precision = np.mean(precision)
    plt.plot(fpr, tpr, color=colors['ensemble'],
             label=f'Ensemble (AUC = {ensemble_auc:.3f}, p = {ensemble_p:.3e}, Precision = {ensemble_avg_precision:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Comparación de Modelos de Diabetes')
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))

    # Crear directorios para los modelos si no existen
    models_dir = os.path.join('app', 'models', 'diabetes')
    os.makedirs(models_dir, exist_ok=True)

    # Guardar la curva ROC
    plt.savefig(os.path.join(models_dir, 'roc_curves.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def calculate_roc_p_value(y_true, y_pred):
    """
    Calcula el p-value para la curva ROC usando el método de bootstrap
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_bootstraps = 1000
    roc_auc_scores = []
    n_samples = len(y_true)

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
        return 1.0

    p_value = (np.sum(np.array(roc_auc_scores) <= 0.5) + 1) / \
        (n_bootstraps + 1)
    return p_value


def train_ensemble(X_train, X_test, y_train, y_test, feature_names):
    """
    Entrena un modelo de ensamble para diabetes usando Random Forest, LGBM y XGBoost
    con parámetros optimizados para predicción de diabetes
    """
    os.makedirs('models/diabetes', exist_ok=True)

    # Analizar el desbalance de clases
    class_counts = np.bincount(y_train)
    minority_class_count = class_counts.min()
    majority_class_count = class_counts.max()
    imbalance_ratio = majority_class_count / minority_class_count

    print(f"\nDistribución de clases original:")
    print(f"Clase mayoritaria: {majority_class_count} muestras")
    print(f"Clase minoritaria: {minority_class_count} muestras")
    print(f"Ratio de desbalance: {imbalance_ratio:.2f}")

    # Ajustar la estrategia de SMOTE según el desbalance
    if imbalance_ratio > 3:
        # Si hay mucho desbalance, usar una estrategia más conservadora
        sampling_strategy = min(
            0.3, (majority_class_count / minority_class_count) / 3)
        print(
            f"\nUsando estrategia de SMOTE conservadora: {sampling_strategy:.2f}")
        try:
            smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train, y_train)
            print("SMOTE aplicado exitosamente")
        except ValueError as e:
            print(f"No se pudo aplicar SMOTE: {str(e)}")
            print("Continuando con datos no balanceados...")
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        # Si el desbalance no es tan severo, usar los datos originales
        print("\nDesbalance no severo, usando datos originales")
        X_train_balanced, y_train_balanced = X_train, y_train

    # Ajustar parámetros según el tamaño del dataset
    n_estimators = min(100, max(50, len(X_train_balanced) // 10))

    # Parámetros para Random Forest
    rf_params = {
        'n_estimators': n_estimators,
        'max_depth': 3,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42
    }

    # Parámetros para LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 8,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'max_depth': 3,
        'verbose': -1,
        'scale_pos_weight': imbalance_ratio  # Ajuste por desbalance
    }

    # Parámetros para XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 1.0,
        'alpha': 1.0,
        'lambda': 1.0,
        'scale_pos_weight': imbalance_ratio  # Ajuste por desbalance
    }

    # Validación cruzada
    # Ajustar splits según la clase minoritaria
    n_splits = min(10, min(np.bincount(y_train)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Random Forest
    rf_model = RandomForestClassifier(**rf_params)
    rf_scores = cross_val_score(rf_model, X_train, y_train,
                                cv=skf, scoring='roc_auc')
    rf_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"Random Forest CV AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_scores = cross_val_score(lgb_model, X_train, y_train,
                                 cv=skf, scoring='roc_auc')
    lgb_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"LightGBM CV AUC: {lgb_scores.mean():.4f} (+/- {lgb_scores.std() * 2:.4f})")

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_scores = cross_val_score(xgb_model, X_train, y_train,
                                 cv=skf, scoring='roc_auc')
    xgb_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"XGBoost CV AUC: {xgb_scores.mean():.4f} (+/- {xgb_scores.std() * 2:.4f})")

    # Predicciones
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
    models_dir = os.path.join('app', 'models', 'diabetes')
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


def generate_diagnostic_table(X_test, y_test, model, original_df):
    """
    Genera una tabla de diagnóstico para los primeros 30 casos de prueba.

    Parameters:
    -----------
    X_test : pd.DataFrame
        Features de prueba
    y_test : pd.Series
        Etiquetas reales de prueba
    model : dict
        Diccionario con los modelos entrenados y pesos
    original_df : pd.DataFrame
        DataFrame original con los valores sin normalizar

    Returns:
    --------
    pd.DataFrame
        Tabla con los diagnósticos
    """
    # Obtener predicciones de cada modelo
    rf_pred = model['rf_model'].predict_proba(X_test)[:, 1]
    lgb_pred = model['lgb_model'].predict_proba(X_test)[:, 1]
    xgb_pred = model['xgb_model'].predict_proba(X_test)[:, 1]

    # Calcular predicción del ensemble
    ensemble_pred = (
        model['weights'][0] * rf_pred +
        model['weights'][1] * lgb_pred +
        model['weights'][2] * xgb_pred
    )

    # Crear DataFrame con los primeros 30 casos
    df_diagnostic = pd.DataFrame()

    # Obtener índices de los primeros 30 casos
    test_indices = X_test.index[:30]

    # Agregar características relevantes con valores originales
    df_diagnostic['HbA1c (%)'] = original_df.loc[test_indices,
                                                 'hba1c'].round(1)
    df_diagnostic['Glucosa (mg/dL)'] = original_df.loc[test_indices,
                                                       'fasting_glucose'].round(1)
    df_diagnostic['IMC'] = original_df.loc[test_indices, 'bmi'].round(1)
    df_diagnostic['Edad'] = original_df.loc[test_indices, 'age'].astype(int)
    df_diagnostic['Triglicéridos (mg/dL)'] = original_df.loc[test_indices,
                                                             'triglycerides'].round(1)

    # Agregar variables de estilo de vida
    df_diagnostic['Actividad Física'] = original_df.loc[test_indices,
                                                        'physical_activity']
    df_diagnostic['Consumo Alcohol'] = original_df.loc[test_indices,
                                                       'alcohol_consumption']
    df_diagnostic['Fumador'] = original_df.loc[test_indices, 'smoker'].map({
                                                                           0: 'No', 1: 'Sí'})
    df_diagnostic['Historia Familiar'] = original_df.loc[test_indices,
                                                         'family_history_diabetes'].map({0: 'No', 1: 'Sí'})

    # Agregar diagnóstico real
    df_diagnostic['Diagnóstico Real'] = y_test.iloc[:30].map(
        {0: 'No Diabético', 1: 'Diabético'})

    # Agregar predicción del modelo
    df_diagnostic['Predicción Modelo'] = (
        ensemble_pred[:30] >= model['threshold']).astype(int)
    df_diagnostic['Predicción Modelo'] = df_diagnostic['Predicción Modelo'].map(
        {0: 'No Diabético', 1: 'Diabético'})

    # Agregar columna de acierto
    df_diagnostic['Acierto'] = (
        df_diagnostic['Diagnóstico Real'] == df_diagnostic['Predicción Modelo'])

    return df_diagnostic


def main():
    # Cargar dataset
    print("Cargando dataset...")
    df = pd.read_csv('app/datasets/final_dataset.csv')
    original_df = df.copy()  # Guardar copia con valores originales

    # Preparar datos
    df, features = prepare_data(df)

    # División de datos
    X = df[features]
    y = df['diabetes']  # Target variable para diabetes

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Entrenar modelo
    print("\nEntrenando modelos...")
    model = train_ensemble(X_train, X_test, y_train, y_test, features)

    # Generar y guardar tabla de diagnóstico
    print("\nGenerando tabla de diagnóstico...")
    diagnostic_table = generate_diagnostic_table(
        X_test, y_test, model, original_df)

    # Crear directorio para resultados si no existe
    results_dir = os.path.join('app', 'results', 'diabetes')
    os.makedirs(results_dir, exist_ok=True)

    # Guardar tabla en CSV
    diagnostic_table.to_csv(os.path.join(
        results_dir, 'diagnostic_table.csv'), index=True)
    print(
        f"Tabla de diagnóstico guardada en: {os.path.join(results_dir, 'diagnostic_table.csv')}")

    print("\nModelos guardados en el directorio 'models/diabetes/'")
    print("Curva ROC guardada como 'models/diabetes/roc_curves.png'")


if __name__ == "__main__":
    main()
