import streamlit as st
import pandas as pd
import os

# Información sobre el dataset
st.markdown("""
# About Dataset
This dataset contains 74,283 records from 20 countries, providing insights into Alzheimer's disease risk factors. It includes demographic, lifestyle, medical, and genetic variables, with a biased distribution to reflect real-world disparities across regions.

This dataset is useful for predictive modeling, epidemiological studies, and healthcare research on Alzheimer’s disease.
""")

# Título de la aplicación
st.title("Predicción de Alzheimer")

# Opción para cargar datos manualmente
uploaded_file = st.file_uploader("Carga tu archivo CSV", type=["csv"])

data_path = "alzheimers_dataset"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Datos cargados correctamente.")
else:
    st.error("Por favor, sube un archivo CSV para continuar.")
    st.stop()

# Mostrar vista previa del DataFrame
st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# Mostrar estadísticas descriptivas
st.subheader("Estadísticas descriptivas")
st.write(df.describe())

# Mostrar información del DataFrame
st.subheader("Información del Dataset")
st.text(df.info())

# Espacio para futuras funciones (visualización, modelos, etc.)
