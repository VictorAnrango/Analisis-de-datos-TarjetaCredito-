import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


df = pd.read_csv('creditcard.csv', sep=',')

print(df.head()) #Muestra las primeras filas para verificar 

print("\n cantidad de fraudes y no fraudes")
print(df["Class"].value_counts()) #Cantidad de fraudes y no fraudes
print(df["Class"].value_counts(normalize=True) * 100) #Porcentaje de fraudes y no fraudes

df.isnull().sum().sum() #Verificar valores nulos

df[df.duplicated()]["Class"].value_counts() #Cantidad de fraudes y no fraudes en los registros duplicados
df[df.duplicated()]["Class"].value_counts().sum()

#Eliminar duplicados 
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} duplicate rows.")

#Eliminamos la columna 'Class' para obtener las características y no dar las repuestas a nuestro modelo a entrenar
X = df.drop('Class', axis=1)   
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Normalizar
# Normalización Min-Max
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print(pd.DataFrame(X_train_scaled).head())

# Balancear clases con SMOTE
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("Distribución original:")
print(y.value_counts())

print("\nDistribución después de SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

#reducimos la informacion lo mas compacta posible, sin perder informacion importante, para que el modelo pueda aprender mejor

# PCA completo
pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_train_resampled)

explained_variance = pca_full.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)

# Gráfica
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker='o'
)

plt.axhline(y=0.95, linestyle='--')

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")

plt.grid(True)

plt.show()

# Número de componentes
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print("Components for 95% variance:", n_components_95)

# PCA final
pca = PCA(n_components=n_components_95)

X_train_pca = pca.fit_transform(X_train_resampled)
X_test_pca = pca.transform(X_test_scaled)

#Logistic Regression
log_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

log_model.fit(X_train_pca, y_train_resampled)

y_pred_log = log_model.predict(X_test_pca)

print("\nLOGISTIC REGRESSION")

print("Accuracy:")
print(accuracy_score(y_test, y_pred_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

#Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train_pca, y_train_resampled)

y_pred_rf = rf_model.predict(X_test_pca)
print("\nRANDOM FOREST")

print("Accuracy:")
print(accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

#ROC

#probabilidades para Logistic Regression
y_prob_log = log_model.predict_proba(X_test_pca)[:, 1]

#probabilidades para Random Forest
y_prob_rf = rf_model.predict_proba(X_test_pca)[:, 1]

#calcular ROC for logistic regression
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
auc_log = roc_auc_score(y_test, y_prob_log)

#calcular ROC for random forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

#graficar ROCplt.figure(figsize=(10, 7))

plt.plot(
    fpr_log,
    tpr_log,
    label=f"Logistic Regression (AUC = {auc_log:.4f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.4f})"
)

# Línea aleatoria
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()