import streamlit as st
import pandas as pd
import numpy as np
import os
import gzip
import pickle
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset
st.title("Análisis del Dataset de Alzheimer")

# Cargar el archivo CSV desde GitHub o local
csv_file = "alzheimers_prediction_dataset.csv"
df = pd.read_csv(csv_file)

# Mostrar información general del dataset
st.sidebar.header("Información del Dataset")
st.sidebar.write(f"Total de registros: {df.shape[0]}")
st.sidebar.write(f"Total de columnas: {df.shape[1]}")

# Descripción de las variables en la barra lateral
st.sidebar.subheader("Descripción de Variables")
column_descriptions = {
    "Country": "País de origen del individuo.",
    "Age": "Edad en años.",
    "Gender": "Género (Masculino/Femenino).",
    "Education Level": "Nivel educativo alcanzado.",
    "BMI": "Índice de Masa Corporal.",
    "Physical Activity Level": "Nivel de actividad física (Bajo, Medio, Alto).",
    "Smoking Status": "Estado de fumador (Nunca, Exfumador, Actual).",
    "Alcohol Consumption": "Consumo de alcohol (Nunca, Ocasional, Frecuente).",
    "Diabetes": "Diagnóstico de diabetes (Sí/No).",
    "Hypertension": "Presión arterial alta (Sí/No).",
    "Cholesterol Level": "Nivel de colesterol (Normal, Alto).",
    "Family History of Alzheimer’s": "Historial familiar de Alzheimer.",
    "Cognitive Test Score": "Puntaje en pruebas cognitivas.",
    "Depression Level": "Nivel de depresión (Bajo, Medio, Alto).",
    "Sleep Quality": "Calidad del sueño (Buena, Regular, Mala).",
    "Dietary Habits": "Hábitos alimenticios saludables o no.",
    "Air Pollution Exposure": "Exposición a contaminación ambiental.",
    "Employment Status": "Estado laboral (Activo/Desempleado).",
    "Marital Status": "Estado civil.",
    "Genetic Risk Factor (APOE-ε4 allele)": "Presencia del alelo APOE-ε4.",
    "Social Engagement Level": "Nivel de interacción social.",
    "Income Level": "Nivel de ingresos (Bajo, Medio, Alto).",
    "Stress Levels": "Nivel de estrés (Bajo, Medio, Alto).",
    "Urban vs Rural Living": "Ubicación de residencia (Urbana/Rural).",
    "Alzheimer’s Diagnosis": "Diagnóstico de Alzheimer (Sí/No)."
}

for col, desc in column_descriptions.items():
    st.sidebar.write(f"**{col}:** {desc}")

# Mostrar vista previa del dataset con barra interactiva
st.subheader("Vista Previa del Dataset")
n = st.slider("Número de filas a mostrar", 1, 100, 5)
st.dataframe(df.head(n))

# Mostrar estadísticas descriptivas
df_info = df.describe(include='all')
st.subheader("Estadísticas Descriptivas")
st.write(df_info)

# Mostrar información general del dataset
df_info_text = str(df.info())
st.text(df_info_text)

# Cargar modelo de predicción
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

# Barra lateral para la predicción
st.sidebar.title("Predicción de Alzheimer")
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

# Obtener valores de entrada numéricos
temp_col = st.sidebar.container()
for feature in numeric_features:
    user_input[feature] = temp_col.number_input(feature, min_value=0, step=1, format="%d")
for feature in continuous_features:
    user_input[feature] = temp_col.number_input(feature, value=0.0, format="%.2f")

# Obtener valores de entrada categóricos
temp_col2 = st.sidebar.container()
for feature in categorical_features:
    if feature in label_encoders:
        user_input[feature] = temp_col2.selectbox(feature, label_encoders[feature].classes_)

# Botón de predicción
if st.sidebar.button("Predecir"):
    if model is None:
        st.sidebar.error("No se puede realizar la predicción porque el modelo no se cargó correctamente.")
    else:
        try:
            df_input = pd.DataFrame([user_input])

            # Aplicar Label Encoding correctamente
            for col in categorical_features:
                if col in label_encoders:
                    if user_input[col] in label_encoders[col].classes_:
                        df_input[col] = label_encoders[col].transform([user_input[col]])[0]
                    else:
                        st.sidebar.error(f"El valor '{user_input[col]}' no está en el conjunto de entrenamiento del LabelEncoder.")
                        st.stop()

            # Convertir todas las columnas numéricas a float32
            df_input = df_input.astype(np.float32)

            # Convertir a array NumPy con la forma correcta
            input_array = df_input.to_numpy().reshape(1, -1)

            # Hacer la predicción
            prediction = np.argmax(model.predict(input_array))
            resultado = "Positivo para Alzheimer" if prediction == 1 else "Negativo para Alzheimer"
            st.sidebar.subheader("Resultado de la Predicción")
            st.sidebar.write(resultado)
        except Exception as e:
            st.sidebar.error(f"Ocurrió un error al hacer la predicción: {str(e)}")
