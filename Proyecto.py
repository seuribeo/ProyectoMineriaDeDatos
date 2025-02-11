import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io  # Para capturar la salida de df.info()
import numpy as np
import gzip
import pickle
from sklearn.preprocessing import LabelEncoder

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

    st.sidebar.title("📌 Descripción de Variables")
    
    descripciones = {
        "Country": "País de origen del paciente.",
        "Age": "Edad del paciente en años.",
        "Gender": "Género del paciente (Masculino/Femenino).",
        "Education Level": "Nivel educativo en años completados.",
        "BMI": "Índice de Masa Corporal (IMC) del paciente.",
        "Physical Activity Level": "Frecuencia de actividad física realizada.",
        "Alzheimer’s Diagnosis": "Diagnóstico de Alzheimer (Sí/No)."
    }
    
    variable_seleccionada = st.sidebar.selectbox("\U0001F4CC Selecciona una variable:", list(descripciones.keys()))
    st.sidebar.write(f"**{variable_seleccionada}:** {descripciones[variable_seleccionada]}")

    st.subheader("📂 Información del Dataset")
    
    st.markdown(f"- **Número de registros:** {df.shape[0]:,}")
    st.markdown(f"- **Número de columnas:** {df.shape[1]}")
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("👀 Vista Previa del Dataset")
    num_rows = st.slider("\U0001F4CC Selecciona el número de filas a mostrar:", 1, 100, 5, 1)
    st.write(df.head(num_rows))

    st.subheader("📊 Estadísticas Descriptivas")
    st.write(df.describe())

    st.subheader("📈 Distribución de Variables Numéricas")
    columna_numerica = st.selectbox("\U0001F4CC Selecciona una variable numérica:", df.select_dtypes(include=['number']).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[columna_numerica], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribución de {columna_numerica}")
    st.pyplot(fig)

    st.subheader("📊 Visualización de Variables Categóricas")
    columna_categorica = st.selectbox("\U0001F4CC Selecciona una variable categórica:", df.select_dtypes(include=['object']).columns)
    fig, ax = plt.subplots()
    df[columna_categorica].value_counts().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(f"Distribución de {columna_categorica}")
    st.pyplot(fig)

    st.sidebar.subheader("\U0001F9E0 Predicción de Alzheimer")
    
    @st.cache_resource
    def load_model():
        with gzip.open("mejor_modelo_redes.pkl.gz", 'rb') as f:
            return pickle.load(f)

    @st.cache_resource
    def load_label_encoders():
        with open("label_encoders.pkl", "rb") as f:
            return pickle.load(f)

    model = load_model()
    label_encoders = load_label_encoders()

    user_input = {}
    
    for feature in descripciones.keys():
        if df[feature].dtype == 'object':
            opciones = df[feature].dropna().unique().tolist()
            user_input[feature] = st.sidebar.selectbox(feature, opciones)
        else:
            min_val, max_val = df[feature].min(), df[feature].max()
            user_input[feature] = st.sidebar.slider(feature, float(min_val), float(max_val), float(min_val))
    
    if st.sidebar.button("Predecir"):
        try:
            for feature in user_input.keys():
                if feature in label_encoders:
                    if user_input[feature] not in label_encoders[feature].classes_:
                        user_input[feature] = 'Unknown'
                    user_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
            
            input_df = pd.DataFrame([user_input])
            st.write("Entrada procesada para predicción:", input_df)
            
            prediccion = model.predict(input_df)[0]
            resultado = "Positivo para Alzheimer" if prediccion == 1 else "Negativo para Alzheimer"
            st.sidebar.write(f"\U0001F4FE **Resultado:** {resultado}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error en la predicción: {e}")

except FileNotFoundError:
    st.error(f"⚠️ El archivo {file_path} no se encontró.")


