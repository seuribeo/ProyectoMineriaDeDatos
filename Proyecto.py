import streamlit as st
import pandas as pd
import os

# Información sobre el dataset
st.markdown("""
# About Dataset
This dataset contains 74,283 records from 20 countries, providing insights into Alzheimer's disease risk factors. It includes demographic, lifestyle, medical, and genetic variables, with a biased distribution to reflect real-world disparities across regions.

This dataset is useful for predictive modeling, epidemiological studies, and healthcare research on Alzheimer’s disease.
""")
# Descargar el dataset desde Kaggle
path = kagglehub.dataset_download("ankushpanday1/alzheimers-prediction-dataset-global")

# Verificar la ruta donde se descargó el dataset
print("Path to dataset files:", path)

# Listar archivos dentro del dataset
print("Archivos en el dataset:", os.listdir(path))

# Cargar el archivo CSV en un DataFrame
# Asegúrate de usar el nombre correcto del archivo dentro de la carpeta descargada
archivo_csv = [f for f in os.listdir(path) if f.endswith(".csv")][0]  # Encuentra el archivo CSV automáticamente
df = pd.read_csv(os.path.join(path, archivo_csv))

