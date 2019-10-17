# Machine Learning
Disciplina IN1102 - Aprendizadem de Máquina, da pós-graduação no CIn-UFPE.

Neste repositório estarão presentes os projetos da disciplina, com códigos e relatórios.

## Professores:
- Francisco de Assis Tenório de Carvalho
- Cleber Zanchettin

## Projetos
### *Image Segmentation*
Aplicação de algoritmos de aprendizagem supervisionada e não supervisionada, sendo a base de dados utilizada a [*Image Segmentation*](http://archive.ics.uci.edu/ml/datasets/image+segmentation) do _UCI Machine Learning Repository_. O particular dessa base é a existência de 2 visualizações, dois tipos de dados, um relativo às cores RGB, e a outra relativa à forma da imagem.

As especificações do projeto, definidas pelo professor Francisco de A. T. de Carvalho, podem ser encontradas [aqui](https://www.cin.ufpe.br/~fatc/AM/Projeto-AM-2019-2.pdf).

#### Parte 1 (aprendizado não-supervisionado)
Essa parte do projeto é relativo a um aprendizado não-supervisionado para geração de partições *fuzzy* e, a partir delas, criar partições *crisp*. Baseia-se no algoritmo "*Variable-wise kernel fuzzy clustering algorithm with kernelization of the metric*". Detalhes desse algoritmo são mostrados no artigo "*Kernel fuzzy c-means with automatic variable weighting*", em que algumas equações implementadas no código são apresentadas no artigo (27, 31 e 32). A avaliação de performance será mensurada pelo índice de Rand, que mede a similaridade entre dois agrupamentos.

#### Parte 2 (aprendizado supevisionado)
Agora partindo para outra abordagem, são considerados os rótulos da base de dados para o aprendizado supervisionado. Os classifadores utilizados são combinados pela regra da soma, a partir dos seguintes classificadores (um para cada _view_ de atributos, RGB e *shape*):
- Classificador bayesiano gaussiano.
- Classificador bayesiano baseado em *k*-vizinhos.

É utilizado o teste não paramétrico de Wilcoxon para comparar os comitês de classificadores.

### Projeto prof. Cleber [a definir]