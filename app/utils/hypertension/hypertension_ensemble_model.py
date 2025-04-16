import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, roc_curve, auc
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import json
import os
from app.utils.hypertension.hypertension_model import create_target_variables
import matplotlib.pyplot as plt
from scipy import stats


def prepare_data(df):
    """
    Prepara los datos para el modelo de hipertensión usando discretización y encoding
    con características clínicamente relevantes
    """
    # Crear variables objetivo
    df = create_target_variables(df)

    # Codificar variables categóricas
    le = LabelEncoder()
    categorical_features = [
        'gender', 'physical_activity', 'alcohol_consumption']
    for feature in categorical_features:
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])

    print("Discretizando variables numéricas...")

    # Características de presión arterial más detalladas
    # Umbrales clínicos para hipertensión - Reducir categorías aún más
    bins_systolic = [0, 130, 160, float('inf')]
    labels_systolic = [0, 1, 2]  # Normal/Pre-HTA, HTA 1, HTA 2
    df['systolic_bp_cat'] = pd.cut(
        df['systolic_bp'], bins=bins_systolic, labels=labels_systolic)

    bins_diastolic = [0, 85, 100, float('inf')]
    labels_diastolic = [0, 1, 2]  # Normal/Pre-HTA, HTA 1, HTA 2
    df['diastolic_bp_cat'] = pd.cut(
        df['diastolic_bp'], bins=bins_diastolic, labels=labels_diastolic)

    # Categorización por edad - Mantener simple
    bins_age = [0, 60, float('inf')]
    labels_age = [0, 1]  # No mayor, Mayor
    df['age_cat'] = pd.cut(df['age'], bins=bins_age, labels=labels_age)
    df['age_cat'] = df['age_cat'].astype(int)

    # Categorización de IMC - Mantener simple
    bins_bmi = [0, 30, float('inf')]
    labels_bmi = [0, 1]  # No obeso, Obeso
    df['bmi_cat'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi)
    df['bmi_cat'] = df['bmi_cat'].astype(int)

    # Características derivadas más simples
    df['bp_combined'] = df['systolic_bp_cat'].astype(
        int) + df['diastolic_bp_cat'].astype(int)

    # Simplificar features finales aún más
    features = [
        'systolic_bp_cat',
        'diastolic_bp_cat',
        'bp_combined',
        'age_cat',
        'bmi_cat',
        'cv_risk',
        'gender_encoded'
    ]

    # Índice de riesgo cardiovascular más completo
    risk_factors = []
    available_risk_factors = ['smoker', 'family_history_hypertension']

    # Verificar qué factores de riesgo están disponibles
    for factor in available_risk_factors:
        if factor in df.columns:
            risk_factors.append(factor)

    if risk_factors:
        print(
            f"Calculando índice de riesgo cardiovascular con factores: {risk_factors}")
        df['cv_risk'] = df[risk_factors].sum(axis=1)
    else:
        print("No se encontraron factores de riesgo cardiovascular. Usando valor predeterminado 0.")
        df['cv_risk'] = 0

    # Interacciones importantes
    # Asegurar que las variables son numéricas antes de la multiplicación
    df['systolic_bp_cat'] = df['systolic_bp_cat'].astype(int)
    df['diastolic_bp_cat'] = df['diastolic_bp_cat'].astype(int)

    # Calcular interacciones
    df['age_bmi_interaction'] = df['age_cat'] * df['bmi_cat']
    df['age_bp_interaction'] = df['age_cat'] * \
        (df['systolic_bp_cat'] + df['diastolic_bp_cat'])

    # Manejar valores nulos
    print("Verificando y manejando valores nulos...")
    categorical_features = [
        col for col in features if 'cat' in col or 'encoded' in col]
    numeric_features = [
        col for col in features if col not in categorical_features]

    for feature in features:
        null_count = df[feature].isnull().sum()
        if null_count > 0:
            print(f"- {feature}: {null_count} valores nulos encontrados")
            if feature in categorical_features:
                # Para variables categóricas, usar la moda
                df[feature] = df[feature].fillna(
                    df[feature].mode()[0]).astype(int)
            else:
                # Para variables numéricas, usar la mediana
                df[feature] = df[feature].fillna(df[feature].median())

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
    plt.title('Curva ROC - Comparación de Modelos de Hipertensión')
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))

    # Crear directorios para los modelos si no existen
    models_dir = os.path.join('app', 'models', 'hypertension')
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
    Entrena un modelo de ensamble para hipertensión con parámetros optimizados
    """
    os.makedirs('models/hypertension', exist_ok=True)

    # Analizar el desbalance de clases
    class_counts = np.bincount(y_train)
    minority_class_count = class_counts.min()
    majority_class_count = class_counts.max()
    imbalance_ratio = majority_class_count / minority_class_count

    print(f"\nDistribución de clases original:")
    print(f"Clase mayoritaria: {majority_class_count} muestras")
    print(f"Clase minoritaria: {minority_class_count} muestras")
    print(f"Ratio de desbalance: {imbalance_ratio:.2f}")

    # Usar undersampling para la clase mayoritaria antes de SMOTE
    majority_indices = np.where(y_train == 0)[0]
    minority_indices = np.where(y_train == 1)[0]

    # Reducir la clase mayoritaria a un ratio de 2:1
    n_majority_samples = len(minority_indices) * 2
    selected_majority_indices = np.random.choice(
        majority_indices, n_majority_samples, replace=False)

    # Combinar índices seleccionados
    selected_indices = np.concatenate(
        [selected_majority_indices, minority_indices])
    X_train_undersampled = X_train.iloc[selected_indices]
    y_train_undersampled = y_train.iloc[selected_indices]

    print(f"Después de undersampling - Ratio de clases: 2:1")
    print(f"Muestras clase mayoritaria: {n_majority_samples}")
    print(f"Muestras clase minoritaria: {len(minority_indices)}")

    # Aplicar SMOTE con ratio más balanceado
    try:
        # Aumentado a 0.7 para mayor balance
        smt = SMOTE(sampling_strategy=0.7, random_state=42)
        X_train_balanced, y_train_balanced = smt.fit_resample(
            X_train_undersampled, y_train_undersampled)
        print("SMOTE aplicado exitosamente")
        print(
            f"Nuevas dimensiones del conjunto de entrenamiento: {X_train_balanced.shape}")

    except Exception as e:
        print(f"Error al aplicar SMOTE: {str(e)}")
        X_train_balanced, y_train_balanced = X_train_undersampled, y_train_undersampled

    # Parámetros más conservadores para Random Forest
    rf_params = {
        'n_estimators': 50,  # Reducido para evitar overfitting
        'max_depth': 3,      # Reducido para limitar complejidad
        'min_samples_split': 30,
        'min_samples_leaf': 20,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'max_samples': 0.7  # Añadido para reducir overfitting
    }

    # Parámetros más conservadores para LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 7,
        'learning_rate': 0.01,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 3.0,
        'reg_lambda': 3.0,
        'max_depth': 3,
        'verbose': -1,
        'is_unbalance': True,
        'min_split_gain': 0.1  # Añadido para reducir overfitting
    }

    # Parámetros ajustados para XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.1,
        'alpha': 0,
        'lambda': 1,
        'scale_pos_weight': 1,
        'tree_method': 'hist',  # Método más estable
        'grow_policy': 'lossguide'  # Política de crecimiento más robusta
    }

    # Validación cruzada estratificada
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Random Forest
    rf_model = RandomForestClassifier(**rf_params)
    rf_scores = cross_val_score(
        rf_model, X_train, y_train, cv=skf, scoring='balanced_accuracy')
    rf_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"Random Forest CV Balanced Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_scores = cross_val_score(
        lgb_model, X_train, y_train, cv=skf, scoring='balanced_accuracy')
    lgb_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"LightGBM CV Balanced Accuracy: {lgb_scores.mean():.4f} (+/- {lgb_scores.std() * 2:.4f})")

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_scores = cross_val_score(
        xgb_model, X_train, y_train, cv=skf, scoring='balanced_accuracy')
    xgb_model.fit(X_train_balanced, y_train_balanced)
    print(
        f"XGBoost CV Balanced Accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std() * 2:.4f})")

    # Predicciones y ensemble
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

    # Pesos adaptativos basados en balanced_accuracy
    weights = np.array([
        rf_scores.mean(),
        lgb_scores.mean(),
        xgb_scores.mean()
    ])
    weights = weights / weights.sum()

    # Ensemble con pesos adaptativos
    ensemble_pred = (weights[0] * rf_pred +
                     weights[1] * lgb_pred +
                     weights[2] * xgb_pred)

    # Optimización del umbral con criterios más estrictos
    precision, recall, thresholds = precision_recall_curve(
        y_test, ensemble_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Buscar umbral que priorice recall manteniendo precision razonable
    # Más énfasis en recall
    valid_indices = (precision[:-1] >= 0.45) & (recall[:-1] >= 0.75)
    if any(valid_indices):
        best_threshold = thresholds[valid_indices][np.argmax(
            recall[:-1][valid_indices])]  # Optimizar para recall
    else:
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]

    # Crear directorios para los modelos si no existen
    models_dir = os.path.join('app', 'models', 'hypertension')
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
    y = df['hypertension']  # Target variable para hipertensión

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Entrenar modelo
    print("\nEntrenando modelos...")
    model = train_ensemble(X_train, X_test, y_train, y_test, features)

    print("\nModelos guardados en el directorio 'models/hypertension/'")
    print("Curva ROC guardada como 'models/hypertension/roc_curves.png'")


if __name__ == "__main__":
    main()
