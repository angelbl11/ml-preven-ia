def create_target_variables(df):
    """
    Crear variables objetivo basadas en los umbrales clínicos
    """
    # Obesidad: IMC > 30
    df['obesity'] = (df['bmi'] > 30).astype(int)

    # Diabetes: Glucosa en ayunas > 126 mg/dL
    df['diabetes'] = (df['fasting_glucose'] > 126).astype(int)

    # Hipertensión: PAS ≥ 140 o PAD ≥ 90
    df['hypertension'] = ((df['systolic_bp'] >= 140) |
                          (df['diastolic_bp'] >= 90)).astype(int)

    return df


def create_feature_interactions(df):
    """
    Crear features derivadas basadas en interacciones conocidas y datos epidemiológicos de México
    """
    # Interacción 1: IMC y Edad (mayor riesgo en adultos jóvenes mexicanos)
    df['age_bmi_risk'] = df['bmi'] * (df['age'] / 30)  # Normalizado a 30 años

    # Interacción 2: IMC y Actividad Física (sedentarismo en México)
    activity_map = {'none': 1.0, 'light': 0.8,
                    'moderate': 0.6, 'frequent': 0.4}
    df['sedentary_risk'] = df['bmi'] * \
        df['physical_activity'].map(activity_map)

    # Interacción 3: IMC y Consumo de Alcohol (patrones mexicanos)
    alcohol_map = {'none': 1.0, 'light': 1.2, 'moderate': 1.4, 'heavy': 1.6}
    df['alcohol_bmi_risk'] = df['bmi'] * \
        df['alcohol_consumption'].map(alcohol_map)

    # Interacción 4: IMC y Tabaquismo (prevalencia en México)
    df['smoking_bmi_risk'] = df['bmi'] * (1 + 0.2 * df['smoker'])

    # Interacción 5: IMC y Historia Familiar (heredabilidad en población mexicana)
    df['family_obesity_risk'] = df['bmi'] * \
        (1 + 0.3 * df['family_history_obesity'])

    # Interacción 6: IMC y Glucosa (resistencia a la insulina)
    df['insulin_resistance_risk'] = df['bmi'] * (df['fasting_glucose'] / 100)

    # Interacción 7: IMC y Presión Arterial (comorbilidad común)
    df['hypertension_risk'] = df['bmi'] * \
        ((df['systolic_bp'] + df['diastolic_bp']) / 200)

    # Interacción 8: IMC y LDL (dislipidemia asociada)
    df['lipid_risk'] = df['bmi'] * (df['ldl'] / 100)

    # Interacción 9: IMC y Triglicéridos (síndrome metabólico)
    df['metabolic_risk'] = df['bmi'] * (df['triglycerides'] / 150)

    # Interacción 10: Género y Edad (diferencias por sexo en México)
    # Mayor riesgo en mujeres mexicanas
    gender_map = {'male': 1.0, 'female': 1.2}
    df['gender_age_risk'] = df['age'] * df['gender'].map(gender_map)

    return df
