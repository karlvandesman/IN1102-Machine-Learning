#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Karl Sousa"
__email__       = "kvms@cin.ufpe.br"

# ****************************
# *** Descrição do Projeto ***
# ****************************
# Este código refere-se a segunda parte do projeto da disciplina de Aprendizagem de Máquina,
# do professor Francisco de A. T. de Carvalho. Os algoritmos utilizados são bayesianos,
# considerando uma abordagem paramétrica (normal multivatiada) e outra não paramétrica (kNN).

# ******************************
# *** Importando bibliotecas ***
# ******************************
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from numpy.linalg import det
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
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
y = dataset.index

# Classes (k é o número de classes diferentes), e ordena alfabeticamente
# classes_num -> número de exemplos de cada classe
classes, classes_num = np.unique(np.sort(dataset.index), return_counts=True)

# Objeto para transformar as labels de numérico <-> categórico
encoding = LabelEncoder()
encoding.classes_ = classes

y = encoding.transform(y)

# ********************************
# *** Definição dos parâmetros ***
# ********************************

L = 2	# Número de views (para o comitê)
n = len(dataset) # número de exemplos do treinamento
k = len(classes) # número de classes

# Número de atributos de cada 'view' (espaço de características diferentes)
nShape = len(atributosShape)
nRGB = len(atributosRGB)

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
PClasse = classes_num/len(dataset)

# Cálculo de parâmetro das normais multivariadas de cada classe
# *** Vetor de médias ***
# Média de vetores de cada classe (média por classe e por atributo)
''' Quando o dataframe é agrupado, ele é apresentado em ordem alfabética.
    Isso pode gerar erros na hora de atribuir valores para as classes.'''
sigmaShape = (dataShape.groupby(dataShape.index).mean()).values
sigmaRGB = (dataRGB.groupby(dataRGB.index).mean()).values

# *** Matriz de covariância ***
# Será considerada a aproximação em que as covariância entre diferentes atributos é zero. Ou seja, os valores não-nulos serão somente os da diagonal, e assumindo um mesmo valor para todos os atributos.
'''não sei se essa forma de cálculo está correta'''
varShape = dataShape.groupby(dataShape.index).var().var(axis=1)
varRGB = dataRGB.groupby(dataRGB.index).var().var(axis=1)

covShape = [ np.eye(np.size(dataShape, 1)) * varShape[i] for i in range(k) ]
covRGB = [ np.eye(np.size(dataRGB, 1)) * varRGB[i] for i in range(k) ]

# Cálculo da densidade de probabilidade diretamente
#dataShapeNorm = [ dataShape.values - sigmaShape[i] for i in range(k) ]
#covShapeInv = 1/det(covShape)
#densShape = [ (2*np.pi)**(-nShape/2) * np.sqrt(det(covShapeInv[i])) * np.exp(-0.5 * np.trace(dataShapeNorm[i]) * covShapeInv[i] * (dataShapeNorm[i])) for i in range(k) ] 

# As densidades de probabilidade são calculadas segundo uma normal multivariada com parâmetros sigma e matriz de covariância
# Cálculo usando a função do SciPy
'''Conferir o uso da função multivariate_normal para o cálculo da densidade de probabilidade'''
densShape = [ multivariate_normal.pdf(dataShape.values, sigmaShape[i], covShape[i]) for i in range(k) ]
densRGB = [ multivariate_normal.pdf(dataRGB.values, sigmaRGB[i], covRGB[i]) for i in range(k) ]

# Plot da densidade de probabilidade (está muito estranha!!!)
#x_axis = range(210)
#plt.plot(dataShape.values, multivariate_normal.pdf(dataShape.values, sigmaShape[0], covShape[0]))
#plt.show()

#densShape = [ multivariate_normal.pdf(dataShape[30*i:30*(i+1)].values, sigmaShape[i], covShape[i]) for i in range(k) ] 
#densRGB = [ multivariate_normal.pdf(dataRGB[30*i:30*(i+1)].values, sigmaRGB[i], covRGB[i]) for i in range(k) ] 

# A probabilidade a posteriori é calculada segundo o teorema de Bayes
evidenciaShape = sum(np.dot(np.matrix.transpose(np.array(densShape)), PClasse))
evidenciaRGB = sum(np.dot(np.matrix.transpose(np.array(densRGB)), PClasse))

PGaussShape = [ (densShape[i] * PClasse[i])/evidenciaShape for i in range(k) ] 
PGaussRGB = [ (densRGB[i] * PClasse[i])/evidenciaRGB for i in range(k) ]

# Agora obtemos o classificador combinado pela regra de soma. Vai ser atribuida a classe que obtiver maior soma.
# A decisão de atribuir uma determinada classe a um exemplo é dada por:
# P(wj) + Pgauss,shape(wj|xk) + Pgauss,rgb(wj|xk) = max_{(r=1)}^7 [ P(wr) + Pgauss,shape(wr|xk) + Pgauss,rgb(wr|xk) ]
# Ou seja, a classe que retornar o maior valor para a soma de 3 componentes: as duas probabilidades a posteriori (shape e view) e a estimativa de máxima verossimilhança da classe (Pwi)
probClf1 = [ (1 - L)*PClasse[i] + PGaussShape[i] + PGaussRGB[i] for i in range(k) ]

# Ajustar dimensões da matriz de probabilidade do classificador
probClf1 = np.vstack(probClf1).T

# Agora deve-se obter o índice (qual classe) retornou o maior valor de probabilidade
y_pred = np.argmax(probClf1, axis=1)
print(y_pred)

#a) Validação cruzada estratificada repetida: "30 times ten-fold"
n_folds = 10
n_repeticoes = 30
seed = 10

rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeticoes, random_state=seed)

# o vetor acuracia vai reunir os 300 valores obtidos pelo RepeatedKFold
acuracia = []

for indice_treino, indice_validacao in rkf.split(X):
    X_treino, X_val = X[indice_treino], X[indice_validacao]
    y_treino, y_val = y[indice_treino], y[indice_validacao]
    
    # Adicionar parte de treinamento/predição do classificador
    ### COMEÇAR CÓDIGO AQUI ###
    # 
    #
    ### TERMINAR CÓDIGO AQUI ###
    
    acuracia.append(accuracy_score(y_val, y_pred))

# Agora obtemos as médias de acurácias de 10 em 10 rodadas
acuraciaKfold = np.asarray([ acuracia[i:i+10].mean() 
                for i in range(0, n_folds*n_repeticoes, 10) ])