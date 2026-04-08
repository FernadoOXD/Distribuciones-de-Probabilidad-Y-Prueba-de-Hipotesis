import streamlit as st
import pandas as pd
import numpy as np 

st.set_page_config(page_title="APP Estadistica", layout="wide")
st.title("Analisis Estadistico con IA")
st.sidebar.header("1. Carga de Datos")
upload_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("Vista previa de los datos: ", df.head())
    variables = df.columns.tolist()
    var_objetivo = st.sidebar.selectbox("Selecciona la variable a analizar", variables)
    datos_selecionados = df[[var_objetivo]]
else:
    st.info("Por favor, sube un archivo CSV para comenzar")
    datos_selecionados = pd.Series(np.random.normal(loc=50, scale=10, size=100), name="Datos de ejemplo")
