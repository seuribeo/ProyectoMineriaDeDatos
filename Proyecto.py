import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io  # Para capturar la salida de df.info()

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

    # **Cantidad de categorías en variables categóricas**
    st.subheader("📌 Variables Categóricas - Cantidad de Categorías")
    categorias_por_variable = df.select_dtypes(include=['object']).nunique()
    st.write(categorias_por_variable)

    # **Gráficos de distribución**
    st.subheader("📈 Distribución de Variables Numéricas")

    # Selector para elegir variable numérica y graficar
    columna_numerica = st.selectbox("📌 Selecciona una variable numérica:", df.select_dtypes(include=['number']).columns)

    # Histograma de la variable seleccionada
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

except FileNotFoundError:
    st.error(f"⚠️ El archivo {file_path} no se encontró. Asegúrate de que está en la misma carpeta que el script.")
