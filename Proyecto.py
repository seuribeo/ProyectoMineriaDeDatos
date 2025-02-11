import streamlit as st
import pandas as pd
import os
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Información sobre el dataset
st.markdown("""
# 📊 Análisis del Dataset de Alzheimer
Este conjunto de datos contiene 74,283 registros de 20 países, proporcionando información sobre los factores de riesgo de la enfermedad de Alzheimer. Incluye variables demográficas, de estilo de vida, médicas y genéticas, con una distribución sesgada para reflejar disparidades reales en distintas regiones.

Es útil para la modelización predictiva, estudios epidemiológicos e investigación en salud sobre la enfermedad de Alzheimer.
""")

# Cargar el dataset
archivo_csv = "alzheimers_prediction_dataset.csv"
df = pd.read_csv(archivo_csv)

# Mostrar información del dataset con opción interactiva en la barra lateral
st.sidebar.header("📌 Información del Dataset")
opcion_variable = st.sidebar.selectbox("Selecciona una variable para describir:", df.columns)
st.sidebar.write(df[opcion_variable].describe())

# Mostrar información general del dataset
st.subheader("🔍 Vista previa del dataset")
num_filas = st.slider("Selecciona el número de filas a visualizar", 1, 100, 5)
st.dataframe(df.head(num_filas))

st.subheader("📊 Estadísticas descriptivas")
st.write(df.describe())

st.subheader("📌 Información general")
st.text(df.info())

# Visualización de gráficos
st.subheader("📈 Distribución de variables")
columna_seleccionada = st.selectbox("Selecciona una variable para graficar:", df.columns)
fig, ax = plt.subplots()
sns.histplot(df[columna_seleccionada], kde=True, ax=ax)
st.pyplot(fig)

# Cargar el modelo y los encoders
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

# Sección de predicción en la barra lateral
st.sidebar.header("🧠 Predicción de Alzheimer")
categorical_features = [
    'Country', 'Gender', 'Smoking Status', 'Alcohol Consumption', 'Diabetes',
    'Hypertension', 'Cholesterol Level', 'Family History of Alzheimer’s',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Urban vs Rural Living', 'Physical Activity Level', 'Depression Level',
    'Sleep Quality', 'Dietary Habits', 'Air Pollution Exposure',
    'Social Engagement Level', 'Income Level', 'Stress Levels'
]

numeric_features = ['Age', 'Education Level', 'Cognitive Test Score']
continuous_features = ['BMI']
user_input = {}

for feature in numeric_features:
    user_input[feature] = st.sidebar.number_input(feature, min_value=0, step=1, format="%d")

for feature in continuous_features:
    user_input[feature] = st.sidebar.number_input(feature, value=0.0, format="%.2f")

for feature in categorical_features:
    if feature in label_encoders:
        user_input[feature] = st.sidebar.selectbox(feature, label_encoders[feature].classes_)

if st.sidebar.button("🔍 Predecir"):
    if model is None:
        st.sidebar.error("No se puede realizar la predicción porque el modelo no se cargó correctamente.")
    else:
        try:
            df_input = pd.DataFrame([user_input])
            for col in categorical_features:
                if col in label_encoders:
                    if user_input[col] in label_encoders[col].classes_:
                        df_input[col] = label_encoders[col].transform([user_input[col]])[0]
                    else:
                        st.sidebar.error(f"El valor '{user_input[col]}' no está en el conjunto de entrenamiento del LabelEncoder.")
                        st.stop()
            df_input = df_input.astype(np.float32)
            input_array = df_input.to_numpy().reshape(1, -1)
            prediction = np.argmax(model.predict(input_array))
            resultado = "Positivo para Alzheimer" if prediction == 1 else "Negativo para Alzheimer"
            st.sidebar.subheader("📌 Resultado de la Predicción")
            st.sidebar.write(resultado)
        except Exception as e:
            st.sidebar.error(f"Ocurrió un error al hacer la predicción: {str(e)}")
