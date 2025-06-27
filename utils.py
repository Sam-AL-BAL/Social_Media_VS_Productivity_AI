import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st


modelo = joblib.load("modelo_entrenado.pkl")
scaler = joblib.load("scaler.pkl")
columnas_entrenamiento = joblib.load("columnas_entrenamiento.pkl")

def preparar_datos_para_modelo(df_usuario):
    X = pd.get_dummies(df_usuario, drop_first=True)
    for col in columnas_entrenamiento:
        if col not in X.columns:
            X[col] = 0
    X = X[columnas_entrenamiento]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(numeric_only=True), inplace=True)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=columnas_entrenamiento)
    return X_scaled

def predecir(df_usuario):
    X = preparar_datos_para_modelo(df_usuario)
    pred = modelo.predict(X).flatten()
    return pred[0]

def comparar_real_vs_predicho(X_test_scaled, Y_test):
    pred = modelo.predict(X_test_scaled).flatten()
    return pd.DataFrame({'Real': Y_test.values, 'Prediccion': pred})

def generar_figura_shap(X_scaled):
    explainer = shap.Explainer(modelo)
    shap_values = explainer(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_scaled, feature_names=columnas_entrenamiento, show=False)
    plt.tight_layout()
    return fig


def menu():
    with st.sidebar:
        st.header("Menú")
        st.page_link(page="predecir.py", label="Predicción")
        st.page_link(page="pages/preprocesamiento.py", label= "Data adicional")
        