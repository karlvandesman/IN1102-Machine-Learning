#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Karl Sousa"
__email__       = "kvms@cin.ufpe.br"

# ****************************
# *** Descrição do Projeto ***
# ****************************
# Este código refere-se a segunda parte do projeto da disciplina de 
# Aprendizagem de Máquina, do professor Francisco de A. T. de Carvalho. Os 
# algoritmos utilizados são bayesianos, considerando uma abordagem paramétrica 
# (normal multivatiada) e outra não paramétrica (kNN).

# ******************************
# *** Importando bibliotecas ***
# ******************************
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

from sklearn.neighbors import NearestNeighbors

# ---------------------
# ** Classificador 2 **
# ---------------------
# Classificador combinado pela regra da soma a partir do classificador bayesiano baseado em k-vizinhos.

# Treinar dois classificadores bayesianos baseados em k-vizinhos, um para cada view.
# Normalizar os dados e usar a distância euclidiana
# Usar conjunto de validação para fixar o k

num_K = 10 	# Número máximo de vizinhos K a ser avaliado

standard_scaler = StandardScaler()
X_escalonado = standard_scaler.fit_transform(X)

for i in num_K:
	### Treinamento ###
	# É considerada a distância euclidiana como padrão
	knnShape = NearestNeighbors(n_neighbors=i)
	knnRGB = NearestNeighbors(n_neighbors=i)
	
	knnShape.fit(X_train[atributosShape])
	knnRGB.fit(X_train[atributosRGB])

	### Validação ###
	indiceKvizShape = knnShape.kneighbors(X_val[atributosShape])
	indiceKvizRGB = knnRGB.kneighbors(Xval[atributosRGB])

	kiShape = [ sum(y_train[indiceKvizShape]==j) for j in range(k) ]
	kiRGB = [ sum(y_train[indiceKvizRGB]==j) for j in range(k) ]

	# A probabilidade é definida como o número de vizinhos de uma classe
	# dividido pelo número total de vizinhos

	PkvizShape = [ kiShape[j]/i for j in range(k) ]
	PkvizRGB = [ kiRGB[j]/i for j in range(k) ]

# Agora obtemos o classificador combinado pela regra de soma.
# Vai ser atribuida a classe que obtiver maior soma das probabilidades.
probClf2 = [ Pclasse[i] + PkvizShape[i] + PkvizRGB[i] for i in range(k) ]
probClf2 = np.vstack(probClf2).T

y_pred = np.argmax(probClf2, axis=1)

#for i in range(20):
#    model = KNeighborsClassifier(n_neighbors = i)
#
#    model.fit(diabetes_X_train, diabetes_y_train)  #fit the model
#    pred=model.predict(diabetes_X_test) #make prediction on test set
#    error = sqrt(mean_squared_error(diabetes_y_test,pred)) #calculate rmse
#    rmse_val.append(error) #store rmse values
#    r2=r2_score(diabetes_y_test,pred)
#    print('RMSE para k= ' , K , 'é:', error, 'e R2 score', r2)
#

# Obter estimativa pontual e intervalo de confiança para o acerto do clf