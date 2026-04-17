import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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

st.header("Prueba de Hipótesis (Prueba Z)")

col1_h, col2_h, col3_h = st.columns(3)
with col1_h:
    h0_mean = st.number_input("Media Hipotética (H0)", value=float(datos_seleccionados.mean()))
with col2_h:
    tipo_prueba = st.selectbox("Tipo de Prueba", ["Bilateral", "Cola Izquierda", "Cola Derecha"])
with col3_h:
    alpha = st.selectbox("Nivel de Significancia (Alpha)", [0.01, 0.05, 0.10], index=1)


media_muestral = datos_seleccionados.mean()
n = len(datos_seleccionados)
std_dev = datos_seleccionados.std() 
error_estandar = std_dev / np.sqrt(n)
z_stat = (media_muestral - h0_mean) / error_estandar


if tipo_prueba == "Bilateral":
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    z_critico = stats.norm.ppf(1 - alpha/2)
elif tipo_prueba == "Cola Izquierda":
    p_value = stats.norm.cdf(z_stat)
    z_critico = stats.norm.ppf(alpha)
else: 
    p_value = 1 - stats.norm.cdf(z_stat)
    z_critico = stats.norm.ppf(1 - alpha)

rechazar_h0 = p_value < alpha


col1, col2 = st.columns(2)
col1.metric("Estadístico Z Calculado", f"{z_stat:.4f}")
col2.metric("P-value", f"{p_value:.4f}")

if rechazar_h0:
    st.error("Decisión: Se RECHAZA la Hipótesis Nula (H0)")
else:
    st.success("Decisión: NO se rechaza la Hipótesis Nula (H0)")

