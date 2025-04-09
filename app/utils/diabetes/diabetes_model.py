def create_target_variables(df):
    """
    Crear variables objetivo basadas en los umbrales clínicos para diabetes
    """
    # Diabetes: Criterios diagnósticos (cualquiera de los siguientes)
    # - Glucosa en ayunas > 126 mg/dL
    # - HbA1c ≥ 6.5%
    df['diabetes'] = ((df['fasting_glucose'] > 126) |
                      (df['hba1c'] >= 6.5)).astype(int)

    # Pre-diabetes: Criterios diagnósticos
    # - Glucosa en ayunas entre 100-125 mg/dL o
    # - HbA1c entre 5.7% y 6.4%
    df['prediabetes'] = ((df['fasting_glucose'].between(100, 125)) |
                         (df['hba1c'].between(5.7, 6.4))).astype(int)

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
    df['glycemic_control'] = df['hba1c'] * (df['fasting_glucose'] / 100)

    # Interacción 2: HbA1c y Edad (riesgo aumenta con la edad)
    df['age_glycemic_risk'] = df['hba1c'] * \
        (df['age'] / 50)  # Normalizado a 50 años

    # Interacción 3: HbA1c e IMC (resistencia a la insulina)
    df['glycemic_bmi_risk'] = df['hba1c'] * \
        (df['bmi'] / 25)  # Normalizado a IMC normal

    # Interacción 4: Actividad Física y Control Glicémico
    activity_map = {'none': 1.0, 'light': 0.8,
                    'moderate': 0.6, 'frequent': 0.4}
    df['glycemic_activity_risk'] = df['hba1c'] * \
        df['physical_activity'].map(activity_map)

    # Interacción 5: Historia Familiar y Control Glicémico
    df['family_glycemic_risk'] = df['hba1c'] * \
        (1 + 0.5 * df['family_history_diabetes'])

    # Interacción 6: IMC y Triglicéridos (síndrome metabólico)
    df['metabolic_syndrome_risk'] = (
        df['bmi'] / 25) * (df['triglycerides'] / 150)

    # Interacción 7: Presión Arterial y Control Glicémico
    df['bp_glycemic_risk'] = (df['hba1c'] / 5.7) * \
        ((df['systolic_bp'] + df['diastolic_bp']) / 200)

    # Interacción 8: Edad y Historia Familiar (riesgo hereditario ajustado por edad)
    df['age_family_risk'] = (df['age'] / 50) * \
        (1 + 0.5 * df['family_history_diabetes'])

    # Interacción 9: Género y Control Glicémico (diferencias por sexo)
    gender_map = {'male': 1.0, 'female': 1.1}
    df['gender_glycemic_risk'] = df['hba1c'] * df['gender'].map(gender_map)

    # Interacción 10: Control Glicémico y Perfil Lipídico
    df['glycemic_lipid_risk'] = (
        df['hba1c'] / 5.7) * ((df['ldl'] / 100) + (df['triglycerides'] / 150))

    return df
