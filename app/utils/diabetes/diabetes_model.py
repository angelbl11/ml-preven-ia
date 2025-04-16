def create_target_variables(df):
    """
    Crear variables objetivo basadas en los umbrales clínicos para diabetes
    """
    # Diabetes: Criterios diagnósticos (cualquiera de los siguientes)
    # - Glucosa en ayunas > 126 mg/dL
    # - HbA1c ≥ 6.5% (si está disponible)
    df['diabetes'] = (df['fasting_glucose'] > 126)
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['diabetes'] = df['diabetes'] | (df['hba1c'] >= 6.5)
    df['diabetes'] = df['diabetes'].astype(int)

    # Pre-diabetes: Criterios diagnósticos
    # - Glucosa en ayunas entre 100-125 mg/dL o
    # - HbA1c entre 5.7% y 6.4% (si está disponible)
    df['prediabetes'] = df['fasting_glucose'].between(100, 125)
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['prediabetes'] = df['prediabetes'] | df['hba1c'].between(5.7, 6.4)
    df['prediabetes'] = df['prediabetes'].astype(int)

    # Riesgo metabólico: Combinación de factores
    df['metabolic_risk'] = ((df['bmi'] > 30) |  # Obesidad
                            (df['systolic_bp'] >= 140) |  # Hipertensión
                            # Triglicéridos altos
                            (df['triglycerides'] > 150)).astype(int)

    return df


def create_feature_interactions(df):
    """
    Crear features derivadas basadas en interacciones conocidas y datos epidemiológicos de México
    para diabetes
    """
    # Interacción 1: HbA1c y Glucosa (correlación principal)
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['glycemic_control'] = df['hba1c'] * (df['fasting_glucose'] / 100)
    else:
        # Si no hay HbA1c, usar solo glucosa normalizada
        df['glycemic_control'] = df['fasting_glucose'] / 100

    # Interacción 2: HbA1c y Edad (riesgo aumenta con la edad)
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['age_glycemic_risk'] = df['hba1c'] * \
            (df['age'] / 50)  # Normalizado a 50 años
    else:
        # Si no hay HbA1c, usar solo edad normalizada
        df['age_glycemic_risk'] = df['age'] / 50

    # Interacción 3: HbA1c e IMC (resistencia a la insulina)
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['glycemic_bmi_risk'] = df['hba1c'] * \
            (df['bmi'] / 25)  # Normalizado a IMC normal
    else:
        # Si no hay HbA1c, usar solo IMC normalizado
        df['glycemic_bmi_risk'] = df['bmi'] / 25

    # Interacción 4: Actividad Física y Control Glicémico
    activity_map = {'none': 1.0, 'light': 0.8,
                    'moderate': 0.6, 'frequent': 0.4}
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['glycemic_activity_risk'] = df['hba1c'] * \
            df['physical_activity'].map(activity_map)
    else:
        # Si no hay HbA1c, usar solo actividad física
        df['glycemic_activity_risk'] = df['physical_activity'].map(
            activity_map)

    # Interacción 5: Historia Familiar y Control Glicémico
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['family_glycemic_risk'] = df['hba1c'] * \
            (1 + 0.5 * df['family_history_diabetes'])
    else:
        # Si no hay HbA1c, usar solo historia familiar
        df['family_glycemic_risk'] = 1 + 0.5 * df['family_history_diabetes']

    # Interacción 6: IMC y Triglicéridos (síndrome metabólico)
    df['metabolic_syndrome_risk'] = (
        df['bmi'] / 25) * (df['triglycerides'] / 150)

    # Interacción 7: Presión Arterial y Control Glicémico
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['bp_glycemic_risk'] = (df['hba1c'] / 5.7) * \
            ((df['systolic_bp'] + df['diastolic_bp']) / 200)
    else:
        # Si no hay HbA1c, usar solo presión arterial
        df['bp_glycemic_risk'] = (df['systolic_bp'] + df['diastolic_bp']) / 200

    # Interacción 8: Edad y Historia Familiar (riesgo hereditario ajustado por edad)
    df['age_family_risk'] = (df['age'] / 50) * \
        (1 + 0.5 * df['family_history_diabetes'])

    # Interacción 9: Género y Control Glicémico (diferencias por sexo)
    gender_map = {'male': 1.0, 'female': 1.1}
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['gender_glycemic_risk'] = df['hba1c'] * df['gender'].map(gender_map)
    else:
        # Si no hay HbA1c, usar solo género
        df['gender_glycemic_risk'] = df['gender'].map(gender_map)

    # Interacción 10: Control Glicémico y Perfil Lipídico
    if 'hba1c' in df.columns and not df['hba1c'].isnull().all():
        df['glycemic_lipid_risk'] = (
            df['hba1c'] / 5.7) * ((df['ldl'] / 100) + (df['triglycerides'] / 150))
    else:
        # Si no hay HbA1c, usar solo perfil lipídico
        df['glycemic_lipid_risk'] = (
            df['ldl'] / 100) + (df['triglycerides'] / 150)

    return df
