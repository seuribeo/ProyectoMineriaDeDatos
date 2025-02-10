import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io  # Para capturar la salida de df.info()

# Configurar estilo de gráficos
sns.set_style("whitegrid")

# Título del análisis
st.markdown("""
# Análisis del Dataset sobre Alzheimer
Este dataset contiene 74,283 registros de 20 países y proporciona información sobre los factores de riesgo de la enfermedad de Alzheimer.  
Incluye variables demográficas, de estilo de vida, médicas y genéticas.
""")

# Cargar el archivo CSV
file_path = "alzheimers_prediction_dataset.csv"  # Asegúrate de que el archivo está en la misma carpeta que el script

try:
    df = pd.read_csv(file_path)

    # **Ventana lateral con descripción de variables**
    st.sidebar.title("Descripción de Variables")
    
    # Diccionario con descripciones de las variables (modifica según corresponda)
    descripciones = {
        "Age": "Edad del paciente.",
        "Gender": "Género del paciente (Masculino/Femenino).",
        "Country": "País de origen del paciente.",
        "Education": "Nivel de educación en años.",
        "PhysicalActivity": "Nivel de actividad física.",
        "Smoking": "Historial de tabaquismo (Sí/No).",
        "Alcohol": "Consumo de alcohol (Sí/No).",
        "GeneticRisk": "Riesgo genético de desarrollar Alzheimer.",
        "CognitiveDecline": "Nivel de deterioro cognitivo.",
        "Diagnosis": "Diagnóstico de Alzheimer (Sí/No)."
    }

    # Selector en la barra lateral para elegir una variable y ver su descripción
    variable_seleccionada = st.sidebar.selectbox("Selecciona una variable para ver su descripción:", list(descripciones.keys()))
    st.sidebar.write(f"**{variable_seleccionada}:** {descripciones[variable_seleccionada]}")

    # **Información general del dataset**
    st.subheader("Información del Dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)  # Capturar la salida de df.info()
    info_df = buffer.getvalue()
    st.text(info_df)  # Mostrar en Streamlit

    # **Previsualización con barra interactiva**
    st.subheader("Vista previa del Dataset")
    num_rows = st.slider("Selecciona el número de filas a mostrar:", min_value=1, max_value=100, value=5, step=1)
    st.write(df.head(num_rows))

    # **Estadísticas descriptivas**
    st.subheader("Estadísticas Descriptivas")
    st.write(df.describe())

    # **Cantidad de categorías en variables categóricas**
    st.subheader("Cantidad de categorías en variables categóricas")
    categorias_por_variable = df.select_dtypes(include=['object']).nunique()
    st.write(categorias_por_variable)

    # **Gráficos de distribución**
    st.subheader("Distribución de Variables")

    # Selector para elegir variable numérica y graficar
    columna_numerica = st.selectbox("Selecciona una variable numérica para visualizar:", df.select_dtypes(include=['number']).columns)

    # Histograma de la variable seleccionada
    fig, ax = plt.subplots()
    sns.histplot(df[columna_numerica], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribución de {columna_numerica}")
    st.pyplot(fig)

    # **Gráfico de barras para variables categóricas**
    st.subheader("Visualización de Variables Categóricas")
    columna_categorica = st.selectbox("Selecciona una variable categórica:", df.select_dtypes(include=['object']).columns)

    fig, ax = plt.subplots()
    df[columna_categorica].value_counts().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(f"Distribución de {columna_categorica}")
    st.pyplot(fig)

except FileNotFoundError:
    st.error(f"El archivo {file_path} no se encontró. Asegúrate de que está en la misma carpeta que el script.")
