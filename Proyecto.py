import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo de gráficos
sns.set_style("whitegrid")

# Información sobre el dataset
st.markdown("""
# About Dataset
This dataset contains 74,283 records from 20 countries, providing insights into Alzheimer's disease risk factors. It includes demographic, lifestyle, medical, and genetic variables, with a biased distribution to reflect real-world disparities across regions.

This dataset is useful for predictive modeling, epidemiological studies, and healthcare research on Alzheimer’s disease.
""")

# Cargar el archivo CSV en un DataFrame
file_path = "alzheimers_prediction_dataset.csv"  # Asegúrate de que el archivo está en la misma carpeta que el script

try:
    df = pd.read_csv(file_path)

    # **Información general del dataset**
    st.subheader("Dataset Information")
    buffer = df.info(buf=None)  # Capturar salida de df.info()
    st.text(buffer)  # Mostrar información en Streamlit

    # **Previsualización con barra interactiva**
    st.subheader("Dataset Preview")
    num_rows = st.slider("Selecciona el número de filas a mostrar:", min_value=1, max_value=100, value=5, step=1)
    st.write(df.head(num_rows))

    # **Estadísticas descriptivas**
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # **Cantidad de categorías para cada variable categórica**
    st.subheader("Categorical Variable Counts")
    categorical_counts = df.select_dtypes(include=['object']).nunique()
    st.write(categorical_counts)

    # **Gráficos de distribución**
    st.subheader("Variable Distributions")

    # Seleccionar variable para visualizar
    selected_column = st.selectbox("Selecciona una variable numérica para graficar:", df.select_dtypes(include=['number']).columns)

    # Gráfico de barras para la variable seleccionada
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribución de {selected_column}")
    st.pyplot(fig)

    # **Gráficos de barras para variables categóricas**
    st.subheader("Categorical Variable Visualization")
    selected_cat_column = st.selectbox("Selecciona una variable categórica:", df.select_dtypes(include=['object']).columns)

    fig, ax = plt.subplots()
    df[selected_cat_column].value_counts().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(f"Distribución de {selected_cat_column}")
    st.pyplot(fig)

except FileNotFoundError:
    st.error(f"El archivo {file_path} no se encontró. Asegúrate de que está en la misma carpeta que el script.")
