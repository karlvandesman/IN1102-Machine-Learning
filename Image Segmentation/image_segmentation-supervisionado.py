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

# Utilizar:
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