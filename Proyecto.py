import streamlit as st
import pandas as pd

# Información sobre el dataset
st.markdown("""
# About Dataset
This dataset contains 74,283 records from 20 countries, providing insights into Alzheimer's disease risk factors. It includes demographic, lifestyle, medical, and genetic variables, with a biased distribution to reflect real-world disparities across regions.

This dataset is useful for predictive modeling, epidemiological studies, and healthcare research on Alzheimer’s disease.
""")

# Cargar el archivo CSV en un DataFrame
file_path = "alzheimers_prediction_dataset.csv"  # Asegúrate de que el archivo está en el mismo directorio del script

try:
    df = pd.read_csv(file_path)

    # Mostrar las primeras filas del dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Mostrar estadísticas descriptivas
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

except FileNotFoundError:
    st.error(f"El archivo {file_path} no se encontró. Asegúrate de que está en la misma carpeta que el script.")


