import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 1. Cargar datos
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
        print(f'Se llenaron los valores faltantes en {col} con la media')
    else:
        df[col].fillna(df[col].median(), inplace=True)
        print(f'Se llenaron los valores faltantes en {col} con la mediana')
        
df = pd.DataFrame(df)

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
modelo = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
modelo.fit(X_train, Y_train)

# 5. Guardar modelo
joblib.dump(modelo, "modelo_entrenado.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(columnas_entrenamiento, "columnas_entrenamiento.pkl")

# 6. Evaluación
y_pred = modelo.predict(X_test)
print("MSE:", mean_squared_error(Y_test, y_pred))
print("MAE:", mean_absolute_error(Y_test, y_pred))
print("R2:", r2_score(Y_test, y_pred))
