import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Configurar estilo de gráficos
sns.set_style("whitegrid")

# Título del análisis
st.markdown("""
# 📊 Análisis del Dataset sobre Alzheimer
Este dataset contiene **74,283 registros** de **20 países** y proporciona información sobre los **factores de riesgo** de la enfermedad de Alzheimer.  
Incluye variables demográficas, de estilo de vida, médicas y genéticas.  

🔹 **Objetivo:** Analizar los factores asociados al Alzheimer y su impacto en diferentes poblaciones.
""")

# Cargar el archivo CSV
file_path = "alzheimers_prediction_dataset.csv"

try:
    df = pd.read_csv(file_path)
    
    # **Ventana lateral con descripción de variables**
    st.sidebar.title("📌 Predicción del Alzheimer")
    
    # Selección de características para la predicción
    features = [
        "Age", "Gender", "Education Level", "BMI", "Physical Activity Level", "Smoking Status", "Alcohol Consumption",
        "Diabetes", "Hypertension", "Cholesterol Level", "Family History of Alzheimer’s", "Cognitive Test Score",
        "Depression Level", "Sleep Quality", "Dietary Habits", "Air Pollution Exposure", "Employment Status",
        "Marital Status", "Genetic Risk Factor (APOE-ε4 allele)", "Social Engagement Level", "Income Level",
        "Stress Levels", "Urban vs Rural Living"
    ]
    
    # Codificar variables categóricas
    encoders = {}
    df_encoded = df.copy()
    
    for col in df.select_dtypes(include=['object']).columns:
        encoders[col] = LabelEncoder()
        df_encoded[col] = encoders[col].fit_transform(df[col])
    
    # Modelo de predicción
    X = df_encoded[features]
    y = df_encoded["Alzheimer’s Diagnosis"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Interfaz de usuario para ingresar valores de predicción
    input_data = {}
    
    for feature in features:
        if df[feature].dtype == 'object':
            unique_values = df[feature].unique()
            input_data[feature] = st.sidebar.selectbox(f"{feature}", unique_values)
        else:
            min_val, max_val = df[feature].min(), df[feature].max()
            input_data[feature] = st.sidebar.slider(f"{feature}", min_val, max_val, min_val)
    
    # Transformar entrada del usuario
    input_df = pd.DataFrame([input_data])
    
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = input_df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else 0)
    
    # Realizar predicción
    prediction = model.predict(input_df)
    prediction_label = "Positivo" if prediction[0] == 1 else "Negativo"
    
    st.sidebar.markdown(f"### 💡 Predicción: **{prediction_label}**")

    # **Información general del dataset**
    st.subheader("📂 Información del Dataset")
    st.markdown(f"- **Número de registros:** {df.shape[0]:,}")
    st.markdown(f"- **Número de columnas:** {df.shape[1]}")
    
    # **Gráficos**
    st.subheader("📈 Distribución de Variables Numéricas")
    num_cols = df.select_dtypes(include=['number']).columns
    selected_num_col = st.selectbox("📌 Selecciona una variable numérica:", num_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_num_col], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribución de {selected_num_col}")
    st.pyplot(fig)
    
    # **Gráfico de barras para variables categóricas**
    st.subheader("📊 Visualización de Variables Categóricas")
    cat_cols = df.select_dtypes(include=['object']).columns
    selected_cat_col = st.selectbox("📌 Selecciona una variable categórica:", cat_cols)
    fig, ax = plt.subplots()
    df[selected_cat_col].value_counts().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(f"Distribución de {selected_cat_col}")
    st.pyplot(fig)
    
except FileNotFoundError:
    st.error(f"⚠️ El archivo {file_path} no se encontró. Asegúrate de que está en la misma carpeta que el script.")
