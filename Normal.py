import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar los datos normalizados desde los archivos Excel
datos_normalizados_minmax = pd.read_excel("D:/@Josefh.QM/UNAP_24_1/Investigacion_mercado/datos_normalizados_minmax.xlsx")
datos_normalizados_zscore = pd.read_excel("D:/@Josefh.QM/UNAP_24_1/Investigacion_mercado/datos_normalizados_zscore.xlsx")
datos_normalizados_decimal = pd.read_excel("D:/@Josefh.QM/UNAP_24_1/Investigacion_mercado/datos_normalizados_decimal.xlsx")

# 1. Análisis Exploratorio de Datos (EDA)
print("Resumen estadístico de los datos con Min-Max Scaling:")
print(datos_normalizados_minmax.describe())

print("\nResumen estadístico de los datos con Z-score Normalization:")
print(datos_normalizados_zscore.describe())

print("\nResumen estadístico de los datos con Decimal Scaling:")
print(datos_normalizados_decimal.describe())

# Graficar histogramas de las variables para cada técnica
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(datos_normalizados_minmax['AlturaNormalizada'], color='blue', alpha=0.5)
plt.title('Histograma de Altura Normalizada (Min-Max Scaling)')
plt.xlabel('Altura Normalizada')
plt.ylabel('Frecuencia')

plt.subplot(1, 3, 2)
plt.hist(datos_normalizados_zscore['AlturaNormalizada'], color='green', alpha=0.5)
plt.title('Histograma de Altura Normalizada (Z-score Normalization)')
plt.xlabel('Altura Normalizada')
plt.ylabel('Frecuencia')

plt.subplot(1, 3, 3)
plt.hist(datos_normalizados_decimal['AlturaNormalizada'], color='red', alpha=0.5)
plt.title('Histograma de Altura Normalizada (Decimal Scaling)')
plt.xlabel('Altura Normalizada')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Gráficos de dispersión de los datos normalizados para cada técnica
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(datos_normalizados_minmax['AlturaNormalizada'], datos_normalizados_minmax['PesoNormalizado'], color='blue', alpha=0.5)
plt.title('Diagrama de Dispersión (Min-Max Scaling)')
plt.xlabel('Altura Normalizada')
plt.ylabel('Peso Normalizado')

plt.subplot(1, 3, 2)
plt.scatter(datos_normalizados_zscore['AlturaNormalizada'], datos_normalizados_zscore['PesoNormalizado'], color='green', alpha=0.5)
plt.title('Diagrama de Dispersión (Z-score Normalization)')
plt.xlabel('Altura Normalizada')
plt.ylabel('Peso Normalizado')

plt.subplot(1, 3, 3)
plt.scatter(datos_normalizados_decimal['AlturaNormalizada'], datos_normalizados_decimal['PesoNormalizado'], color='red', alpha=0.5)
plt.title('Diagrama de Dispersión (Decimal Scaling)')
plt.xlabel('Altura Normalizada')
plt.ylabel('Peso Normalizado')

plt.tight_layout()
plt.show()

# 2. Modelado Estadístico (Regresión Lineal)
def entrenar_modelo_regresion(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regresion = LinearRegression()
    regresion.fit(X_train, y_train)
    y_pred = regresion.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

X_minmax = datos_normalizados_minmax['AlturaNormalizada'].values.reshape(-1, 1)
y_minmax = datos_normalizados_minmax['PesoNormalizado'].values

X_zscore = datos_normalizados_zscore['AlturaNormalizada'].values.reshape(-1, 1)
y_zscore = datos_normalizados_zscore['PesoNormalizado'].values

X_decimal = datos_normalizados_decimal['AlturaNormalizada'].values.reshape(-1, 1)
y_decimal = datos_normalizados_decimal['PesoNormalizado'].values

mse_minmax = entrenar_modelo_regresion(X_minmax, y_minmax)
mse_zscore = entrenar_modelo_regresion(X_zscore, y_zscore)
mse_decimal = entrenar_modelo_regresion(X_decimal, y_decimal)

print("\nError cuadrático medio (MSE) del modelo de regresión lineal con Min-Max Scaling:", mse_minmax)
print("Error cuadrático medio (MSE) del modelo de regresión lineal con Z-score Normalization:", mse_zscore)
print("Error cuadrático medio (MSE) del modelo de regresión lineal con Decimal Scaling:", mse_decimal)

# 3. Comparación de Modelos (Regresión Logística)
def entrenar_modelo_logistico(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regresion = LogisticRegression()
    regresion.fit(X_train, y_train)
    y_pred = regresion.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    return precision

objetivo = np.where(datos_normalizados_minmax['AlturaNormalizada'] > 0.5, 1, 0)

precision_minmax = entrenar_modelo_logistico(X_minmax, objetivo)
precision_zscore = entrenar_modelo_logistico(X_zscore, objetivo)
precision_decimal = entrenar_modelo_logistico(X_decimal, objetivo)

print("\nPrecisión del modelo de regresión logística con Min-Max Scaling:", precision_minmax)
print("Precisión del modelo de regresión logística con Z-score Normalization:", precision_zscore)
print("Precisión del modelo de regresión logística con Decimal Scaling:", precision_decimal)

# 4. Interpretación final
print("\nInterpretación de Resultados:")
print("El modelo de regresión lineal con Min-Max Scaling tiene un MSE de", mse_minmax)
print("El modelo de regresión lineal con Z-score Normalization tiene un MSE de", mse_zscore)
print("El modelo de regresión lineal con Decimal Scaling tiene un MSE de", mse_decimal)

print("\nEl modelo de regresión logística con Min-Max Scaling tiene una precisión de", precision_minmax)
print("El modelo de regresión logística con Z-score Normalization tiene una precisión de", precision_zscore)
print("El modelo de regresión logística con Decimal Scaling tiene una precisión de", precision_decimal)

print("\nConclusiones:")
print("La técnica de normalización que mejor se comporta en términos de MSE para la regresión lineal y precisión para la regresión logística es:")
print("Min-Max Scaling para la regresión lineal y Z-score Normalization para la regresión logística.")
