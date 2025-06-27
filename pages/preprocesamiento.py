import utils
import streamlit as st
import pandas as pd
from utils import comparar_real_vs_predicho
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from utils import generar_figura_shap

utils.menu()
st.title("Información de datos") 

#Cargar el archivo
ruta = 'social_media_vs_productivity.csv'
df  = pd.read_csv(ruta, encoding = 'utf-8')
#Detectar valores faltantes
faltantes = df.isnull()

#Contar valores faltantes por columnas
DatosFaltantes = df.isnull().sum()

columnas_filtradas = DatosFaltantes[DatosFaltantes > 0]

#columnas_filtradas

#Imputación condicional
for col in columnas_filtradas.index:
    skewness= df[col].skew() #Calcular el coeficiente de asimetría de los valores en la columna
    if abs(skewness) < 0.5: #Comprobar el valor absoluto del coeficiente de asimetría es menor a 0.5
        df[col].fillna(df[col].mean(), inplace= True)

    else:
        df[col].fillna(df[col].median(), inplace=True)


st.subheader("DataFrame")
st.dataframe(df.head())

# 2. Separar variables
X = df.drop(columns=['actual_productivity_score'])
Y = df['actual_productivity_score']

# 3. Transformaciones
X = pd.get_dummies(X, drop_first=True)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(numeric_only=True), inplace=True)

columnas_entrenamiento = X.columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Entrenamiento
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

#Comparativa de valores reales y predichos
st.subheader("Comparar valores reales vs predichos")
st.dataframe(comparar_real_vs_predicho(X_test, Y_test))


#Figura SHAP

st.subheader("Importancia de características (SHAP)")

X_scaled = utils.preparar_datos_para_modelo(df)  # o X_test_scaled si estás haciendo pruebas

with st.spinner("Generando gráfico SHAP..."):
    fig = generar_figura_shap(X_scaled)
    st.pyplot(fig)






