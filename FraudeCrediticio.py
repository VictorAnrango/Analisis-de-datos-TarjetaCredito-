import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


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

# Para escalar los datos y mas facil de entender 
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled.head())

#reducimos la informacion lo mas compacta posible, sin perder informacion importante, para que el modelo pueda aprender mejor

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print(explained_variance, '\n')
print(cumulative_variance)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.95, linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
print("Mostrando gráfica...")
plt.show()

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {n_components_95}")

#Creacion del nuevo dataset con las componentes principales seleccionadas
pca_95 = PCA(n_components=n_components_95)

X_pca_95 = pca_95.fit_transform(X_scaled)
X_pca_95 = pd.DataFrame(
    X_pca_95,
    columns=[f"PC{i+1}" for i in range(n_components_95)]
)
print(X_pca_95)

#reducimos a 2D la informacion solo para visualizarla
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca_2, columns=["PC1", "PC2"])
pca_df["Class"] = y.values
print(pca_df.head())

normal_pca_2 = pca_df[pca_df["Class"] == 0]
fraud_pca_2 = pca_df[pca_df["Class"] == 1]

plt.figure(figsize=(8, 5))
plt.scatter(
    normal_pca_2["PC1"],
    normal_pca_2["PC2"],
    alpha=0.4,
    label="Normal",
    s=10
)
plt.scatter(
    fraud_pca_2["PC1"],
    fraud_pca_2["PC2"],
    alpha=0.8,
    label="Fraud",
    s=2
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Visualization: Normal vs Fraud")
plt.legend()
plt.grid(True)
plt.show()

# creacion de t-SNE

#Mezclamos PCA con class
pca_with_class = X_pca_95.copy()
pca_with_class["Class"] = y.values

#Se toman los datos de fraude 
fraud_data = pca_with_class[pca_with_class["Class"] == 1]

#Se toma una muestra de los datos normales para balancear el dataset
normal_data = pca_with_class[pca_with_class["Class"] == 0].sample(
    n=10000,
    random_state=42
)

#Crea un nuevo dataset pero mezcla todo par que no exista un patron 
tsne_subset = pd.concat([normal_data, fraud_data], axis=0)

# Shuffle rows
tsne_subset = tsne_subset.sample(frac=1, random_state=42)

# Split features and labels
X_tsne_subset = tsne_subset.drop("Class", axis=1)
y_tsne_subset = tsne_subset["Class"]

print("t-SNE subset shape:", X_tsne_subset.shape)
print("Class distribution:")
print(y_tsne_subset.value_counts())

#inicio del t-SNE para reducir la informacion a 2D y vizualizar 

start_time = time.time()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
    verbose=1
)

X_tsne = tsne.fit_transform(X_tsne_subset)

end_time = time.time()

print("t-SNE output shape:", X_tsne.shape)
print("Time taken in minutes:", (end_time - start_time) / 60)

# Create a DataFrame for visualization
tsne_df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])

# Add labels only for coloring and interpretation
tsne_df["Class"] = y_tsne_subset.values

print("t-SNE DataFrame shape:", tsne_df.shape)
print("Class distribution:")
print(tsne_df["Class"].value_counts())

tsne_df.head()

plt.figure(figsize=(10, 6))

plt.scatter(
    tsne_df[tsne_df["Class"] == 0]["TSNE1"],
    tsne_df[tsne_df["Class"] == 0]["TSNE2"],
    alpha=0.4,
    s=10,
    label="Normal"
)

plt.scatter(
    tsne_df[tsne_df["Class"] == 1]["TSNE1"],
    tsne_df[tsne_df["Class"] == 1]["TSNE2"],
    alpha=0.9,
    s=25,
    label="Fraud"
)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization: Normal vs Fraud")
plt.legend()
plt.grid(True)
plt.show()


min_samples = 10

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_tsne_subset)

distances, indices = neighbors_fit.kneighbors(X_tsne_subset)

k_distances = np.sort(distances[:, min_samples - 1])

plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{min_samples}th nearest neighbor distance")
plt.title("k-Distance Plot for Choosing DBSCAN eps")
plt.grid(True)
plt.show()
