# ============================================================
# PROYECTO ANALÍTICA DE DATOS - CRISP DM
# Dataset: Wine Quality Red
# ============================================================

# ============================================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO
# ============================================================

"""
OBJETIVO DEL NEGOCIO:
Analizar las características químicas del vino tinto para predecir
su calidad utilizando modelos de Machine Learning.

OBJETIVOS ESPECÍFICOS:
1. Identificar variables importantes que afectan la calidad.
2. Limpiar y preparar los datos.
3. Entrenar modelos predictivos.
4. Evaluar precisión de los modelos.
5. Guardar el modelo para despliegue futuro.
"""

# ============================================================
# IMPORTAR LIBRERÍAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# ============================================================
# FASE 2: COMPRENSIÓN DE LOS DATOS
# ============================================================

# Cargar dataset
df = pd.read_csv("winequality-red.csv", sep=';')

# Mostrar primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Información general
print("\nInformación del dataset:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Variables del dataset
print("\nVariables:")
print(df.columns)

# ============================================================
# IDENTIFICAR VARIABLES CLAVE
# ============================================================

# Variable objetivo
target = 'quality'

# Variables predictoras
features = df.drop(target, axis=1)

# ============================================================
# VISUALIZACIÓN DE ESTADÍSTICAS
# ============================================================

# Histograma de calidad
plt.figure(figsize=(8,5))
sns.countplot(x='quality', data=df)
plt.title("Distribución de Calidad del Vino")
plt.show()

# Matriz de correlación
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()

# ============================================================
# FASE 3: PREPARACIÓN DE DATOS
# ============================================================

# ------------------------------------------------------------
# LIMPIEZA DE DATOS - Missing Values
# ------------------------------------------------------------

print("\nValores faltantes:")
print(df.isnull().sum())

# Imputación de valores faltantes
imputer = SimpleImputer(strategy='mean')

X = df.drop('quality', axis=1)
y = df['quality']

X_imputed = imputer.fit_transform(X)

# ------------------------------------------------------------
# NORMALIZACIÓN - Normalizer
# ------------------------------------------------------------

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X_imputed)

# ------------------------------------------------------------
# TRANSFORMACIONES - Math Formula
# ------------------------------------------------------------

# Crear nueva variable
df['alcohol_acidity_ratio'] = df['alcohol'] / df['fixed acidity']

print("\nNueva variable creada:")
print(df[['alcohol', 'fixed acidity', 'alcohol_acidity_ratio']].head())

# ============================================================
# FASE 4: MODELADO
# ============================================================

# ------------------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# MODELO 1 - ÁRBOL DE DECISIÓN
# ------------------------------------------------------------

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree.fit(X_train, y_train)

# Predicciones
y_pred_tree = decision_tree.predict(X_test)

# ------------------------------------------------------------
# MODELO 2 - RANDOM FOREST
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

random_forest.fit(X_train, y_train)

# Predicciones
y_pred_rf = random_forest.predict(X_test)

# ============================================================
# FASE 5: EVALUACIÓN
# ============================================================

# ------------------------------------------------------------
# PRECISIÓN DEL ÁRBOL DE DECISIÓN
# ------------------------------------------------------------

accuracy_tree = accuracy_score(y_test, y_pred_tree)

print("\n===================================")
print("ÁRBOL DE DECISIÓN")
print("===================================")

print(f"Accuracy: {accuracy_tree:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_tree))

# Matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_tree),
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Matriz de Confusión - Árbol de Decisión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# ------------------------------------------------------------
# PRECISIÓN DEL RANDOM FOREST
# ------------------------------------------------------------

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\n===================================")
print("RANDOM FOREST")
print("===================================")

print(f"Accuracy: {accuracy_rf:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_rf))

# Matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf),
            annot=True,
            fmt='d',
            cmap='Greens')

plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# ============================================================
# AJUSTE DE PARÁMETROS
# ============================================================

print("\nImportancia de variables:")

importance = random_forest.feature_importances_

feature_importance = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importance
})

feature_importance = feature_importance.sort_values(
    by='Importancia',
    ascending=False
)

print(feature_importance)

# Visualización
plt.figure(figsize=(10,6))
sns.barplot(
    x='Importancia',
    y='Variable',
    data=feature_importance
)

plt.title("Importancia de Variables")
plt.show()

# ============================================================
# FASE 6: DESPLIEGUE
# ============================================================

# ------------------------------------------------------------
# GUARDAR MODELO
# ------------------------------------------------------------

joblib.dump(random_forest, 'modelo_random_forest.pkl')

print("\nModelo guardado correctamente.")

# ------------------------------------------------------------
# CARGAR MODELO
# ------------------------------------------------------------

modelo_cargado = joblib.load('modelo_random_forest.pkl')

print("Modelo cargado correctamente.")

# ------------------------------------------------------------
# EJEMPLO DE PREDICCIÓN
# ------------------------------------------------------------

ejemplo = X_test[0].reshape(1, -1)

prediccion = modelo_cargado.predict(ejemplo)

print(f"\nPredicción ejemplo: Calidad = {prediccion[0]}")

# ============================================================
# INTEGRACIÓN CON APIs O BASES DE DATOS
# ============================================================


from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

modelo = joblib.load('modelo_random_forest.pkl')

@app.route('/')
def home():
    return "API de predicción de vinos funcionando"

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json['data']

    prediction = modelo.predict([data])

    return jsonify({
        'prediccion': int(prediction[0])
    })

app.run(debug=True, port=8080)


# ============================================================
# CONCLUSIONES
# ============================================================

"""
1. Se aplicó la metodología CRISP-DM completa.
2. Se realizó limpieza y normalización de datos.
3. Se entrenaron modelos de Machine Learning.
4. Random Forest obtuvo mejor precisión.
5. El modelo quedó listo para despliegue.
"""
