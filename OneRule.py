import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.classifier import OneRClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


nombre_csv= input("\nNombre del archivo: ")
num_de_bins= int(input("\nNumero de bins: "))
dataset = pd.read_csv(nombre_csv,encoding= 'latin1', low_memory=False)
clase=dataset['Species']
x_train,X_test,y_train,Y_test= train_test_split(dataset, clase, test_size=0.2, random_state=0)
#tabla de la prueba con la clase
prueba_species= pd.concat([X_test, Y_test], axis=1)
prueba_species= prueba_species.reset_index(drop=True)
#tabla del test con la clase
test_species= pd.concat([x_train, y_train], axis=1)
test_species= test_species.reset_index(drop=True)


#cuartiles trian
SpalLengthCm=pd.cut(x_train['SepalLengthCm'], bins=num_de_bins, labels=False)
SpalWidthCm=pd.cut(x_train['SepalWidthCm'], bins=num_de_bins, labels=False)
PetalLengthCm=pd.cut(x_train['PetalLengthCm'], bins=num_de_bins, labels=False)
PetalWidthCm=pd.cut(x_train['PetalWidthCm'], bins=num_de_bins, labels=False)

discreto_train= pd.concat([SpalLengthCm, SpalWidthCm, PetalLengthCm, PetalWidthCm], axis=1)
#print("test discretizado: \n",discreto)
#cuartiles prueba
SpalLengthCm=pd.cut(X_test['SepalLengthCm'], bins=num_de_bins, labels=False)
SpalWidthCm=pd.cut(X_test['SepalWidthCm'], bins=num_de_bins, labels=False)
PetalLengthCm=pd.cut(X_test['PetalLengthCm'], bins=num_de_bins, labels=False)
PetalWidthCm=pd.cut(X_test['PetalWidthCm'], bins=num_de_bins, labels=False)

discreto_test= pd.concat([SpalLengthCm, SpalWidthCm, PetalLengthCm, PetalWidthCm], axis=1)


#Convertir a numpy
discreto_test_print= discreto_test.copy()
discreto_train=discreto_train.to_numpy()
discreto_test=discreto_test.to_numpy()

y_train=y_train.to_numpy()
Y_test=Y_test.to_numpy()

# OneR
# Entrenar el clasificador One Rule
one_rule_classifier = OneRClassifier()
one_rule_classifier.fit(discreto_train, y_train)

#print("\nTabla de prueba con datos discretizados:\n", discreto_test_print)

SepalLengthCmColumna= discreto_test_print['SepalLengthCm']

# Hacer predicciones en los datos de prueba
y_pred = one_rule_classifier.predict(discreto_test)

# Imprimir la regla generada por el clasificador One Rule
y_pred_dt= pd.DataFrame(y_pred)

printclass= pd.concat([SepalLengthCmColumna, y_pred_dt], axis=1, ignore_index=True)

#print("\nTabla de prueba con clase y prediccion:\n", y_pred_dt, "\n---------------------\n", SepalLengthCmColumna)
print("\nLa regla generada: ",one_rule_classifier.prediction_dict_)




# Calcular la precisión del clasificador
accuracy = accuracy_score(Y_test, y_pred)
print(f'Precisión del clasificador One Rule: {accuracy * 100:.2f}%\n')



print("Test discretizado: \n",discreto_test_print)