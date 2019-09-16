	#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Karl Sousa"
__email__       = "kvms@cin.ufpe.br"

# ****************************
# *** Descrição do Projeto ***
# ****************************
# Este código refere-se a primeira parte do projeto da disciplina de Aprendizagem de Máquina,
# do professor Francisco de A. T. de Carvalho. O código baseia-se no algoritmo
# "Variable-wise kernel fuzzy clustering algorithm with kernelization of the metric". Detalhes
# desse algoritmo são mostrados no artigo "Kernel fuzzy c-means with automatic variable 
# weighting", em que algumas equações referenciadas aqui são mostradas no artigo.

# ******************************
# *** Importando bibliotecas ***
# ******************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import adjusted_rand_score
 
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
dicClassesInv = dict(map(reversed, dicClasses.items()))

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

# **********************************
# *** Implementação do algoritmo ***
# **********************************

# Parâmetros:
# c=7, m=1.6, T=150, epsilon=10^-10

# *** Atualização dos protótipos dos grupos ***
# Equação (27)

# v_{ij} = \sum_{k=1}^n (u_{ik})^m K(x_{kj}, v_{ij})x_{kj} / ( \sum_{k=1}^n (u_{ik})^m K(x_{kj}, v_{ij}) )

# *** Atualização dos pesos de relevância ***
# Equação (31)


# *** Atualização do grau de pertinência ***
# Equação (32)

# **********************
# *** Partição Crisp ***
# **********************


# ********************************
# *** Índice de Rand corrigido ***
# ********************************

# *** Índice de Rand entre as partições à priori para os resultados obtidos com cada view ***

# *** Índice de Rand entre as partições obtidas com cada view ***