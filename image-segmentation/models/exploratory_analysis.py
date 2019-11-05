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

#%%********************************
# *** Importação de bibliotecas ***
# *********************************

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

#Fazendo o carregamento dos dados diretamente do UCI Machine Learning     
urlTreino = "http://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"

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