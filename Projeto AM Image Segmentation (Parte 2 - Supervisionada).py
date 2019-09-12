#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Karl Sousa"
__email__       = "kvms@cin.ufpe.br"

# ******************************
# *** Importando bibliotecas ***
# ******************************
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from scipy.stats import wilcoxon

# Algoritmos de aprendizagem
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
 
# *****************************
# *** Carregamento database ***
# *****************************

#Fazendo o carregamento dos dados diretamente do UCI Machine Learning     
urlTreino = "http://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
urlTeste = "http://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test"

dataset = pd.read_csv(urlTreino, skiprows=2)
datasetTeste = pd.read_csv(urlTeste, skiprows=2)

atributosShape = dataset.columns[0:9]
atributosRGB = dataset.columns[9:]

dataShape = dataset[atributosShape].copy()  
dataRGB = dataset[atributosRGB].copy()  

# Divisão dos atributos e classe
X = dataset.values
Y = dataset.index

# ********************************
# *** Definição dos parâmetros ***
# ********************************

# n é o número de exemplos no treinamento
n = len(dataset)
# Classes (k é o número de classes diferentes)
classes = dataset.index.unique().values
k = len(classes)

# Número de atributos de cada 'view' (espaço de características diferentes)
nShape = len(atributosShape)
nRGB = len(atributosRGB)

# Criando dicionário para as classes (string -> int)
dicClasses = { '%s'% classes[i]: i for i in range(k) }

# ****************************************
# *** Características da base de dados ***
# ****************************************

print("Dimensões do dataset com característica da forma:")
print(dataShape.shape)
print()

print("Quantidade de valores únicos por atributo:")
print(dataset.nunique())
print()

print("Dimensões do dataset de teste:")
print(datasetTeste.shape)
print()

print("Primeiros dados:")
print(dataset.head(5))

print("Estatísica descritiva:")
print(dataset.describe())

# *************************
# *** Pré-processamento ***
# *************************

# Aqui vemos que o atributo Region-pixel-count possui somente um valor, então não adiciona informação
#dataset = dataset.drop(columns=['REGION-PIXEL-COUNT'])

# *****************************************
# *** Implementação dos classificadores ***
# *****************************************

# ---------------------
# ** Classificador 1 **
# ---------------------
# Classificador combinado pela regra da soma a partir do classificador bayesiano gaussiano (um para cada view).
# Regra de classificação:

# Usar a estimativa de máxima verossimilhança

# *** Probabilidade a priori das classes ***
# (nº de exemplos de cada classse/número total de exemplos)
PClasse = dataset.index.value_counts()/len(dataset)

# Cálculo de parâmetro das normais multivariadas de cada classe
# *** Vetor de médias ***
# Média de vetores de cada classe (média por classe e por atributo)
sigmaShape = dataShape.groupby(dataShape.index).mean()
sigmaRGB = dataRGB.groupby(dataRGB.index).mean()

# *** Matriz de covariância ***
# Será considerada a aproximação em que as covariância entre diferentes atributos é zero. Ou seja, os valores não-nulos serão somente os da diagonal, e assumindo um mesmo valor para todos os atributos.
'''não sei se essa forma de cálculo está correta'''
varShape = dataShape.groupby(dataShape.index).var().var(axis=1)
varRGB = dataRGB.groupby(dataRGB.index).var().var(axis=1)

covShape = [ np.eye(np.size(dataShape, 1)) * varShape[i] for i in range(k) ]
covRGB = [ np.eye(np.size(dataRGB, 1)) * varRGB[i] for i in range(k) ]

# A probabilidade a posteriori é calculada segundo uma normal multivariada com parâmetros sigma matriz de covariância (matriz identidade)
'''Erro no cálculo das probabilidades'''
PPostShape = [ multivariate_normal.pdf(dataShape.values, sigmaShape[i], covShape[i]) for i in range(k) ] 
PPostRGB = [ multivariate_normal.pdf(dataRGB.values, sigmaRGB[i], covRGB[i]) for i in range(k) ] 

# A probabilidade a posteriori é calculada segundo uma normal multivariada com parâmetros sigma matriz de covariância (matriz identidade)
'''Conferir somatório no denominador (resultado do produto de matrizes)'''
PGaussShape = [ (PPostShape[i] * PClasse[i])/(np.dot(PPostShape, PClasse)) for i in range(k) ] 
PGaussRGB = [  (PPostRGB * PClasse)/(np.dot(PPostRGB, PClasse)) for i in range(k) ]

# Agora obtemos o classificador combinado pela regra de soma. Vai ser atribuida a classe que obtiver maior soma.
# A decisão de atribuir uma determinada classe a um exemplo é dada por:
# P(wj) + Pgauss,shape(wj|xk) + Pgauss,rgb(wj|xk) = max_{(r=1)}^7 [ P(wr) + Pgauss,shape(wr|xk) + Pgauss,rgb(wr|xk) ]
# Ou seja, a classe que retornar o maior valor para a soma de 3 componentes: as duas probabilidades a priori (shape e view) e a estimativa de máxima verossimilhança da classe (Pwi)
probClf1 = [ PClasse[i] + PGaussShape[i] + PGaussRGB[i] for i in range(k) ]

# Para obter o maior valor das colunas de um array, pode-se usar 
# np.amax(array, axis=0)

#a) Validação cruzada estratificada repetida: "30 times ten-fold"
#rkf = RepeatedKFold(n_splits=10, n_repeats=30)
#for train_index, test_index in rkf.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]

# ---------------------
# ** Classificador 2 **
# ---------------------
# Classificador combinado pela regra da soma a partir do classificador bayesiano baseado em k-vizinhos.

# Treinar dois classificadores bayesianos baseados em k-vizinhos, um para cada view.
# Normalizar os dados e usar a distância euclidiana
# Usar conjunto de validação para fixar o k

#scaler = StandardScaler()
#Xscaled = scaler.fit_transform(X)
#
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
## Agora obtemos o classificador combinado pela regra de soma. Vai ser atribuida a classe que obtiver maior soma.
#probClf2 = [ Pclasse[i] + PkvizShape[i] + PkvizRGB[i] for i in range(k) ]

#b) Obter estimativa pontual e intervalo de confiança para o acerto de cada clf

#c) Wilcoxon signed-ranks test (teste não paramétrico) para comparar clfs
# Scipy possui a função que implementa o Wilcoxon
# É o teste recomendado quando os dados não possuem distribuição normal. Ele é baseado na diferença entre as duas condições que estão sendo avaliadas.
# 
# Hipótese nula (Ho): A diferença entre os pares segue uma distribuição simétrica em torno do zero.
# Hipótese alternativa (Ha): A diferença não segue uma distribuição simétrica em torno do zero.
#
# São assumidas 3 condições para se aplicar o teste:
# - A variável dependente precisa ser contínua
# - As observações pares são aleatoriamente e independentemente selecionadas
# - Os pares de observação vem da mesma população

#wilcoxon(primeiraCondição, segundaCondição)
