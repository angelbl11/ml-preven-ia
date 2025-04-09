def create_target_variables(df):
    """
    Crear variables objetivo basadas en los umbrales clínicos para hipertensión
    """
    # Hipertensión: Criterios diagnósticos
    # - PAS ≥ 140 mmHg o PAD ≥ 90 mmHg
    df['hypertension'] = ((df['systolic_bp'] >= 140) |
                          (df['diastolic_bp'] >= 90)).astype(int)

    # Pre-hipertensión: PAS 120-139 mmHg o PAD 80-89 mmHg
    df['prehypertension'] = ((df['systolic_bp'].between(120, 139)) |
                             (df['diastolic_bp'].between(80, 89))).astype(int)

    # Riesgo cardiovascular: Combinación de factores
    df['cardiovascular_risk'] = ((df['bmi'] > 30) |  # Obesidad
                                 (df['ldl'] > 130) |  # LDL alto
                                 # Triglicéridos altos
                                 (df['triglycerides'] > 150) |
                                 (df['smoker'] == 1)).astype(int)  # Tabaquismo

    return df


def create_feature_interactions(df):
    """
    Crear features derivadas basadas en interacciones conocidas y datos epidemiológicos de México
    para hipertensión
    """
    # Interacción 1: Presión arterial sistólica y diastólica (carga total)
    df['bp_load'] = (df['systolic_bp'] / 120) * (df['diastolic_bp'] / 80)

    # Interacción 2: Presión arterial y Edad (riesgo aumenta con la edad)
    df['age_bp_risk'] = ((df['systolic_bp'] / 120) +
                         (df['diastolic_bp'] / 80)) * (df['age'] / 50)

    # Interacción 3: Presión arterial e IMC (impacto de la obesidad)
    df['bp_bmi_risk'] = ((df['systolic_bp'] / 120) +
                         (df['diastolic_bp'] / 80)) * (df['bmi'] / 25)

    # Interacción 4: Actividad Física y Presión Arterial
    activity_map = {'none': 1.0, 'light': 0.8,
                    'moderate': 0.6, 'frequent': 0.4}
    df['bp_activity_risk'] = ((df['systolic_bp'] / 120) + (df['diastolic_bp'] / 80)) * \
        df['physical_activity'].map(activity_map)

    # Interacción 5: Historia Familiar y Presión Arterial
    df['family_bp_risk'] = ((df['systolic_bp'] / 120) + (df['diastolic_bp'] / 80)) * \
        (1 + 0.4 * df['family_history_hypertension'])

    # Interacción 6: Sodio y Presión Arterial (si está disponible)
    if 'sodium_intake' in df.columns:
        sodium_map = {'low': 0.8, 'moderate': 1.0, 'high': 1.3}
        df['sodium_bp_risk'] = ((df['systolic_bp'] / 120) + (df['diastolic_bp'] / 80)) * \
            df['sodium_intake'].map(sodium_map)

    # Interacción 7: Tabaquismo y Presión Arterial
    df['smoking_bp_risk'] = ((df['systolic_bp'] / 120) + (df['diastolic_bp'] / 80)) * \
        (1 + 0.3 * df['smoker'])

    # Interacción 8: Alcohol y Presión Arterial
    alcohol_map = {'none': 1.0, 'light': 1.1, 'moderate': 1.3, 'heavy': 1.5}
    df['alcohol_bp_risk'] = ((df['systolic_bp'] / 120) + (df['diastolic_bp'] / 80)) * \
        df['alcohol_consumption'].map(alcohol_map)

    # Interacción 9: Género y Presión Arterial (diferencias por sexo)
    gender_map = {'male': 1.1, 'female': 1.0}  # Mayor riesgo en hombres
    df['gender_bp_risk'] = ((df['systolic_bp'] / 120) + (df['diastolic_bp'] / 80)) * \
        df['gender'].map(gender_map)

    # Interacción 10: Presión Arterial y Perfil Lipídico
    df['bp_lipid_risk'] = ((df['systolic_bp'] / 120) + (df['diastolic_bp'] / 80)) * \
        ((df['ldl'] / 100) + (df['triglycerides'] / 150))

    return df
