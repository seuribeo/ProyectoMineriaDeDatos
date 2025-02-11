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

    # **Sección de predicción en la barra lateral**
    st.sidebar.subheader("🔮 Predicción de Alzheimer")
    
    @st.cache_resource
    def load_model():
        filename = "mejor_modelo_redes.pkl.gz"
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    @st.cache_resource
    def load_label_encoders():
        encoder_file = "label_encoders.pkl"
        with open(encoder_file, "rb") as f:
            encoders = pickle.load(f)
        return encoders

    model = load_model()
    label_encoders = load_label_encoders()

    # Definir características categóricas y numéricas
    categorical_features = list(label_encoders.keys())
    numeric_features = ['Age', 'Education Level', 'Cognitive Test Score']
    continuous_features = ['BMI']
    user_input = {}

    # Obtener valores de entrada numéricos
    for feature in numeric_features:
        user_input[feature] = st.sidebar.number_input(feature, min_value=0, step=1, format="%d")

    for feature in continuous_features:
        user_input[feature] = st.sidebar.number_input(feature, value=0.0, format="%.2f")

    # Obtener valores de entrada categóricos
    for feature in categorical_features:
        user_input[feature] = st.sidebar.selectbox(feature, label_encoders[feature].classes_)

    if st.sidebar.button("Predecir"):
        df_input = pd.DataFrame([user_input])
        for col in categorical_features:
            df_input[col] = label_encoders[col].transform([user_input[col]])[0]
        df_input = df_input.astype(np.float32)
        input_array = df_input.to_numpy().reshape(1, -1)
        prediction = np.argmax(model.predict(input_array))
        resultado = "Positivo para Alzheimer" if prediction == 1 else "Negativo para Alzheimer"
        st.sidebar.subheader("Resultado de la Predicción")
        st.sidebar.write(resultado)

except FileNotFoundError:
    st.error(f"⚠️ El archivo {file_path} no se encontró. Asegúrate de que está en la misma carpeta que el script.")

