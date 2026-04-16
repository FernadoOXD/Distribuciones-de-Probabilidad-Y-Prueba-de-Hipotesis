import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="APP Estadistica", layout="wide")
st.title("Analisis Estadistico con IA")
st.sidebar.header("1. Carga de Datos")
upload_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("Vista previa de los datos: ", df.head())
    variables = df.columns.tolist()
    var_objetivo = st.sidebar.selectbox("Selecciona la variable a analizar", variables)
    datos_seleccionados = df[var_objetivo]
else:
    st.info("Por favor, sube un archivo CSV para comenzar")
    datos_seleccionados = pd.Series(np.random.normal(loc=50, scale=10, size=100), name="Datos de ejemplo")
    
st.divider()

st.header("Visualización de Distribuciones")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(datos_seleccionados, kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Histograma y KDE")

sns.boxplot(x=datos_seleccionados, ax=axes[1], color="lightgreen")
axes[1].set_title("Boxplot (Detección de Outliers)")

st.pyplot(fig)
st.subheader("Análisis de la Distribución")
normal = st.radio("¿La distribución parece normal?", ("Sí", "No"), horizontal=True)
sesgo = st.radio("¿Hay sesgo?", ("Sin sesgo evidente", "Sesgo Positivo (derecha)", "Sesgo Negativo (izquierda)"))
outliers = st.radio("¿Observas outliers?", ("Sí", "No"), horizontal=True)

st.divider()



