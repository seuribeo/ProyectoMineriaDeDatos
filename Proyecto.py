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
file_path = "alzheimers_prediction_dataset.csv"  # Asegúrate de que el archivo está en la misma carpeta que el script

try:
    df = pd.read_csv(file_path)

    # **Ventana lateral con descripción de variables**
    st.sidebar.title("📌 Descripción de Variables")
    
    # Diccionario con descripciones detalladas
    descripciones = {
        "Country": "País de origen del paciente.",
        "Age": "Edad del paciente en años.",
        "Gender": "Género del paciente (Masculino/Femenino).",
        "Education Level": "Nivel educativo en años completados.",
        "BMI": "Índice de Masa Corporal (IMC) del paciente.",
        "Physical Activity Level": "Frecuencia de actividad física realizada.",
        "Smoking Status": "Historial de tabaquismo (Fumador/No fumador).",
        "Alcohol Consumption": "Consumo de alcohol (Sí/No).",
        "Diabetes": "Diagnóstico de diabetes (Sí/No).",
        "Hypertension": "Diagnóstico de hipertensión arterial (Sí/No).",
        "Cholesterol Level": "Clasificación del nivel de colesterol (Alto/Medio/Bajo).",
        "Family History of Alzheimer’s": "Antecedentes familiares de Alzheimer (Sí/No).",
        "Cognitive Test Score": "Puntaje obtenido en pruebas cognitivas.",
        "Depression Level": "Grado de depresión diagnosticado.",
        "Sleep Quality": "Calidad del sueño reportada.",
        "Dietary Habits": "Hábitos alimenticios del paciente.",
        "Air Pollution Exposure": "Nivel de exposición a la contaminación del aire.",
        "Employment Status": "Situación laboral actual (Empleado, Desempleado, Jubilado, etc.).",
        "Marital Status": "Estado civil del paciente.",
        "Genetic Risk Factor (APOE-ε4 allele)": "Presencia del alelo APOE-ε4 (Sí/No).",
        "Social Engagement Level": "Nivel de interacción social.",
        "Income Level": "Nivel de ingresos económicos.",
        "Stress Levels": "Nivel de estrés reportado.",
        "Urban vs Rural Living": "Ubicación de residencia (Urbano/Rural).",
        "Alzheimer’s Diagnosis": "Diagnóstico de Alzheimer (Sí/No)."
    }

    # Selector en la barra lateral para elegir una variable y ver su descripción
    variable_seleccionada = st.sidebar.selectbox("📌 Selecciona una variable:", list(descripciones.keys()))
    st.sidebar.write(f"**{variable_seleccionada}:** {descripciones[variable_seleccionada]}")

    # **Información general del dataset**
    st.subheader("📂 Información del Dataset")
    
    # Mostrar el número de registros y columnas
    st.markdown(f"- **Número de registros:** {df.shape[0]:,}")
    st.markdown(f"- **Número de columnas:** {df.shape[1]}")

    # Mostrar la cantidad de variables categóricas y numéricas
    num_categoricas = df.select_dtypes(include=['object']).shape[1]
    num_numericas = df.select_dtypes(include=['number']).shape[1]
    st.markdown(f"- **Variables categóricas:** {num_categoricas}")
    st.markdown(f"- **Variables numéricas:** {num_numericas}")

    # Información de tipos de datos
    st.subheader("📋 Tipos de Datos y Valores Nulos")
    buffer = io.StringIO()
    df.info(buf=buffer)  # Capturar la salida de df.info()
    info_df = buffer.getvalue()
    st.text(info_df)  # Mostrar en Streamlit

    # **Previsualización con barra interactiva**
    st.subheader("👀 Vista Previa del Dataset")
    num_rows = st.slider("📌 Selecciona el número de filas a mostrar:", min_value=1, max_value=100, value=5, step=1)
    st.write(df.head(num_rows))

    # **Estadísticas descriptivas**
    st.subheader("📊 Estadísticas Descriptivas")
    st.write(df.describe())

    # **Gráficos de distribución**
    st.subheader("📈 Distribución de Variables Numéricas")
    columna_numerica = st.selectbox("📌 Selecciona una variable numérica:", df.select_dtypes(include=['number']).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[columna_numerica], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribución de {columna_numerica}")
    st.pyplot(fig)

    # **Gráfico de barras para variables categóricas**
    st.subheader("📊 Visualización de Variables Categóricas")
    columna_categorica = st.selectbox("📌 Selecciona una variable categórica:", df.select_dtypes(include=['object']).columns)
    fig, ax = plt.subplots()
    df[columna_categorica].value_counts().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(f"Distribución de {columna_categorica}")
    st.pyplot(fig)

# **Sección de Predicción en la Barra Lateral**
st.sidebar.subheader("🧠 Predicción de Alzheimer")

# Cargar modelo y label encoders
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

# Diccionario de entrada del usuario
user_input = {}

for feature in descripciones.keys():
    if df[feature].dtype == 'object':  # Si es categórica
        opciones = df[feature].dropna().unique().tolist()
        user_input[feature] = st.sidebar.selectbox(feature, opciones)
    else:  # Si es numérica
        min_val = df[feature].min()
        max_val = df[feature].max()
        if pd.notna(min_val) and pd.notna(max_val):  # Si tiene valores válidos
            user_input[feature] = st.sidebar.slider(feature, float(min_val), float(max_val), float(min_val))
        else:  # Si no hay un rango definido
            user_input[feature] = st.sidebar.text_input(feature)

# Botón de predicción
if st.sidebar.button("Predecir"):
    try:
        # Convertir datos categóricos con los label encoders
        for feature in user_input.keys():
            if feature in label_encoders:
                user_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        
        # Crear DataFrame con la entrada del usuario
        input_df = pd.DataFrame([user_input])
        
        # Realizar la predicción
        prediccion = model.predict(input_df)[0]
        resultado = "Positivo para Alzheimer" if prediccion == 1 else "Negativo para Alzheimer"
        
        st.sidebar.write(f"🧾 **Resultado:** {resultado}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error en la predicción: {e}")
