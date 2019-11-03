# *Image Segmentation*
Aplicação de algoritmos de aprendizagem supervisionada e não supervisionada, sendo a base de dados utilizada a [*Image Segmentation*](http://archive.ics.uci.edu/ml/datasets/image+segmentation) do _UCI Machine Learning Repository_. Para cada exemplo da base de dados, já se tem uma extração de features, que se dividem em "2 visualizações", dois conjuntos de atributos, um relativo às cores RGB, e a outra relativa à forma da imagem.

As especificações do projeto, definidas pelo professor Francisco de A. T. de Carvalho, podem ser encontradas [aqui](https://www.cin.ufpe.br/~fatc/AM/Projeto-AM-2019-2.pdf).

## Parte 1 (aprendizado não-supervisionado)
Essa parte do projeto é relativo a um aprendizado não-supervisionado para geração de partições *fuzzy* e também, a partir delas, criar partições *crisp*. Baseia-se no algoritmo "*Variable-wise kernel fuzzy clustering algorithm with kernelization of the metric*". Detalhes desse algoritmo são mostrados no artigo "[*Kernel fuzzy c-means with automatic variable weighting*](https://www.sciencedirect.com/science/article/pii/S0165011413002054)", em que algumas equações implementadas no código são apresentadas no artigo (27, 31 e 32). A avaliação de performance será mensurada pelo índice de Rand, que mede a similaridade entre dois agrupamentos.

## Parte 2 (aprendizado supevisionado)
Agora partindo para outra abordagem, são considerados os rótulos da base de dados para o aprendizado supervisionado. Os classifadores utilizados são combinados pela regra da soma, a partir dos seguintes classificadores (um para cada _view_ de atributos, RGB e *shape*):
- Classificador bayesiano gaussiano.
- Classificador bayesiano baseado em *k*-vizinhos.

É utilizado o teste não paramétrico de Wilcoxon para comparar os comitês de classificadores.
