import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Importar el dataset con pandas
nombre_csv= input("nombre del archivo: ")
dataset = pd.read_csv(nombre_csv,encoding= 'latin1', low_memory=False)
clase=dataset[['Species']]
columna_x= input("nombre de la columna a eliminar: ")
datos= dataset.drop([columna_x], axis=1)
x_train,X_test,Y_train,Y_test= train_test_split(datos, clase, test_size=0.2, random_state=0)

#implementar DBSCAN
distancia_dada= float(input("distancia: "))
cantidad_minima= int(input("cantidad minima de puntos: "))
db = DBSCAN(eps=distancia_dada, min_samples=cantidad_minima)
db.fit(x_train)
#obtener los labels
labels = db.fit_predict(datos)
a=np.unique(labels)

#imprimir los labels
for cluster_id in np.unique(labels):
    if cluster_id == -1:
        continue  # Ignorar puntos ruido
    cluster_indices = np.where(labels == cluster_id)
    cluster_data = datos.iloc[cluster_indices]
    print(f"Cluster {cluster_id + 1}:")
    print(cluster_data)



#converir a 2d
pca = PCA(n_components=2)
pca.fit(datos)

pca = pca.transform(datos)

#graficar
plt.scatter(pca[:,0], pca[:,1], c=labels, cmap='Paired')
plt.title("DBSCAN")
plt.show()
