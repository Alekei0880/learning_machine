import numpy as np
import random
import pandas as pd

from math import dist



# Importar el dataset con pandas
nombre_csv= input("nombre del archivo")
dataset = pd.read_csv(nombre_csv,encoding= 'latin1', low_memory=False)
dc=dataset[['Species']]	

datos= dataset.drop(['Species', 'Id'], axis=1)

# Convertir el dataset en una matriz
dt = datos.values.tolist()
dc = dc.values.tolist()
#sacar 15 registros aleatorios

###indices de registros de prueba###
indice=[]
a=list(range(0,len(dt)))
indice=random.sample(a,15)
#print("indices de registros de prueba: ",indice)
###registros de prueba####
prueba=[]
for i in indice:
    prueba.append(dt[i-1])
    dt.pop(i-1)
#print("Registros de prueba: ",prueba) 








#print("datos sin la prueba:* ", dt)
#print("indices de registros de prueba: ",indice)
# primeros centroide
centroides=[]
for i in range(0,3):
    centroides.append(dt[i])
#print("Centroides: ",centroides)

K= int(input("numero de centroides: "))
#funcion para calcular la distancia euclidiana


ds=dt
p=prueba
arreglo = np.array(ds)
Prueba = np.array(p)

renglones= arreglo.shape[0]
columnas= arreglo.shape[1]
centroides = np.zeros((K, columnas))
centroidesn = np.zeros((K, columnas))
cuenta = np.zeros((K))
distancia= np.zeros((K))

for i in range(K):
    centroides[i][:]= arreglo[i][:]

converge= False

while converge != True:
    for i in arreglo:
        for j in range(K):
            distancia[j]= dist(i, centroides[j][:])
        clas = np.where(distancia == min(distancia))[0]
        clase = clas[0]
        centroidesn[clase][:] += i
        cuenta[clase] += 1
    print("centroide actual: \n", centroides)
    print("datos: ", cuenta)
    print("\n")
    print("centroide nuevo: \n", centroidesn)
    print("converge: ", converge)
    print("\n")

    for h in range(K):
        centroidesn[h][:] = centroidesn[h][:] / cuenta[h]
    if np.array_equal(centroides, centroidesn)== True:
        converge = True
    else:
        centroides = centroidesn.copy()
        centroidesn = np.zeros((K, columnas))
        cuenta = np.zeros((K))
    


print("centroide actual: \n", centroides)
print("indices: ", cuenta)
print("\n")
print("centroide nuevo: \n", centroidesn)
print("converge: ", converge)
print("\n")


# Prueba
prueba_df = pd.DataFrame(
    columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    data=prueba
)
prueba_df['Species'] = [dc[i - 1][0] for i in indice]

print("\n Indices de prueba: ", indice)
print("\n")

#asignacion de centroide 
for i in range(len(prueba)):
    for j in range(K):
        distancia[j]= dist(prueba[i], centroides[j][:])
    clas = np.where(distancia == min(distancia))[0]
    clase = clas[0]
    prueba_df.at[i, 'Centroide'] = clase 

print(prueba_df)