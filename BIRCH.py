import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Importar el dataset con pandas
nombre_csv = input("Nombre del archivo: ")
dataset = pd.read_csv(nombre_csv, encoding='latin1', low_memory=False)
columna_x = input("Nombre de la columna a eliminar: ")
datos = dataset.drop([columna_x], axis=1)
#num_clusters = int(input("Número de clusters: "))
radio = float(input("Radio del subárbol: "))  # Agregamos esta línea para el radio
num_ramas = int(input("Número de ramas: "))


# Implementar Birch con los hiperparámetros personalizados
bir = Birch( threshold=radio, branching_factor=num_ramas)

# Obtener los labels
labels = bir.fit_predict(datos)
np.unique(labels)


for cluster_id in np.unique(labels):
    cluster_data = datos[labels == cluster_id]
    print(f"Cluster {cluster_id + 1}:")
    print(cluster_data)

# Convertir a 2D
pca = PCA(n_components=2)
pca.fit(datos)
pca = pca.transform(datos)

# Graficar
plt.scatter(pca[:, 0], pca[:, 1], c=labels, cmap='viridis')
plt.title("BIRCH")
plt.show()
