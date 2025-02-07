import streamlit as st
import pandas as pd
import os
import opendatasets as od

# Información sobre el dataset
st.markdown("""
# About Dataset
This dataset contains 74,283 records from 20 countries, providing insights into Alzheimer's disease risk factors. It includes demographic, lifestyle, medical, and genetic variables, with a biased distribution to reflect real-world disparities across regions.

This dataset is useful for predictive modeling, epidemiological studies, and healthcare research on Alzheimer’s disease.
""")

# Título de la aplicación
st.title("Predicción de Alzheimer")

# Descargar dataset automáticamente desde Kaggle
kaggle_url = "https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global"
data_path = "alzheimers_dataset"

if not os.path.exists(data_path):
    with st.spinner("Descargando dataset de Kaggle..."):
        od.download(kaggle_url)

# Buscar el archivo CSV dentro del dataset descargado
csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
if csv_files:
    df = pd.read_csv(os.path.join(data_path, csv_files[0]))
    st.success("Dataset descargado y cargado correctamente.")
else:
    st.error("No se encontró un archivo CSV en el dataset descargado.")
    st.stop()

# Mostrar opciones de estadísticas descriptivas
st.subheader("Exploración de Datos")
option = st.selectbox("Selecciona una opción de análisis:", ["Vista previa", "Estadísticas descriptivas", "Información del dataset"])

if option == "Vista previa":
    st.subheader("Vista previa del DataFrame")
    st.dataframe(df.head())
elif option == "Estadísticas descriptivas":
    st.subheader("Estadísticas descriptivas")
    st.write(df.describe())
elif option == "Información del dataset":
    st.subheader("Información del Dataset")
    st.text(df.info())

