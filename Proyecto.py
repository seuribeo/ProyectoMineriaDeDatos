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

# Ruta del dataset (asumiendo que ya está descargado en el entorno de ejecución)
data_path = "alzheimers_dataset"
csv_file = "alzheimers_data.csv"  # Ajustar al nombre correcto del archivo en el dataset

if os.path.exists(os.path.join(data_path, csv_file)):
    df = pd.read_csv(os.path.join(data_path, csv_file))
    st.success("Dataset cargado correctamente.")
else:
    st.error("No se encontró el archivo CSV. Asegúrate de que el dataset está en la ruta correcta.")
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
    st.text(str(df.info()))

