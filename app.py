import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import google.generativeai as genai

st.set_page_config(page_title="APP Estadistica", layout="wide")
st.title("Analisis Estadistico con IA")

st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

st.sidebar.header("2. Asistente de IA")
api_key = st.sidebar.text_input("Ingresa tu API Key de Gemini", type="password")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos: ", df.head())
    variables = df.columns.tolist()
    var_objetivo = st.sidebar.selectbox("Selecciona la variable a analizar", variables)
    datos_seleccionados = df[var_objetivo]
else:
    st.info("Por favor, sube un archivo CSV para comenzar")
    datos_seleccionados = pd.Series(np.random.normal(loc=50, scale=10, size=100), name="Datos de ejemplo")
    
st.divider()

st.header("Visualización de Distribuciones")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

sns.histplot(datos_seleccionados, kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Histograma y KDE")

sns.boxplot(x=datos_seleccionados, ax=axes[1], color="lightgreen")
axes[1].set_title("Boxplot (Detección de Outliers)")

st.pyplot(fig)
st.subheader("Análisis de la Distribución")
normal = st.radio("¿La distribución parece normal?", ("Sí", "No"), horizontal=True)
sesgo = st.radio("¿Hay sesgo?", ("Sin sesgo evidente", "Sesgo Positivo (derecha)", "Sesgo Negativo (izquierda)"), horizontal=True)
outliers = st.radio("¿Observas outliers?", ("Sí", "No"), horizontal=True)

with st.expander("Verificar respuestas matemáticamente"):
    valor_sesgo = datos_seleccionados.skew()
    if valor_sesgo > 0.5:
        texto_sesgo = f"Sesgo Positivo (Cola a la derecha) con valor de {valor_sesgo:.2f}"
    elif valor_sesgo < -0.5:
        texto_sesgo = f"Sesgo Negativo (Cola a la izquierda) con valor de {valor_sesgo:.2f}"
    else:
        texto_sesgo = f"Aproximadamente Simétrica (Sin sesgo evidente) con valor de {valor_sesgo:.2f}"
        
    Q1 = datos_seleccionados.quantile(0.25)
    Q3 = datos_seleccionados.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    cantidad_outliers = ((datos_seleccionados < limite_inferior) | (datos_seleccionados > limite_superior)).sum()
    
    st.write(f"- **Diagnóstico de Sesgo:** La computadora detecta que es una distribución **{texto_sesgo}**.")
    if cantidad_outliers > 0:
        st.write(f"- **Diagnóstico de Outliers:** Se detectaron matemáticamente **{cantidad_outliers}** datos atípicos (outliers).")
    else:
        st.write("- **Diagnóstico de Outliers:** No se detectaron datos atípicos.")

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

st.header("Asistente de Inteligencia Artificial")

if st.button("Analizar resultados con IA"):
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Actúa como un experto en estadística. Se realizó una prueba Z con los siguientes parámetros:
        - Media de la muestra = {media_muestral:.2f}
        - Media bajo H0 = {h0_mean}
        - Nivel de significancia (Alfa) = {alpha}
        - Tipo de prueba = {tipo_prueba}
        - Valor Z calculado = {z_stat:.2f}
        - P-value calculado = {p_value:.4f}
        
        ¿Se rechaza H0? Explica la decisión basándote estrictamente en la evidencia estadística. 
        Mantén la respuesta concisa, profesional.
        """
        
        with st.spinner("Gemini está analizando los datos..."):
            try:
                respuesta = model.generate_content(prompt)
                st.info(respuesta.text)
            except Exception as e:
                st.error(f"Error de conexión con la API: {e}")
    else:
        st.warning("Por favor, ingresa tu API Key en la barra lateral para usar el asistente.")