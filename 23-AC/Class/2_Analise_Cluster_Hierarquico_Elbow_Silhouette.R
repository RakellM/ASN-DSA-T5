########################################
#
#   CHAMANDO BIBLIOTECAS IMPORTANTES
#
########################################

library(tidyverse) #pacote para manipulacao de dados
library(cluster) #algoritmo de cluster
library(dendextend) #compara dendogramas
library(factoextra) #algoritmo de cluster e visualizacao
library(fpc) #algoritmo de cluster e visualizacao
library(gridExtra) #para a funcao grid arrange
library(readxl) #carregar dados
library(NbClust) #ajuda na silhueta

########################################
#
#   ENTENDENDO ELBOW E SILLHOUETTE
#
########################################

#LEITURA DOS DADOS
alunos_pap2 <- read.table("dados/alunos_pap2.csv", sep = ";", header = T, dec = ",")
View(alunos_pap2)
rownames(alunos_pap2) <- alunos_pap2[,1]
alunos_pap2 <- alunos_pap2[,-1]

#CALCULANDO MATRIZ DE DISTANCIAS
d2 <- dist(alunos_pap2, method = "euclidean")
d2

#DEFININDO O CLUSTER A PARTIR DO METODO ESCOLHIDO
hc_2 <- hclust(d2, method = "average" )

#DESENHANDO O DENDOGRAMA
plot(hc_2, cex = 0.6, hang = -1)

#BRINCANDO COM O DENDOGRAMA PARA 2 GRUPOS
rect.hclust(hc_2, k = 9)

#marcando grupos
grupos <- cutree(hc_2, k = 2)

#VERIFICANDO ELBOW
elbow <- fviz_nbclust(alunos_pap2, FUN = hcut, method = "wss", hc_method = "average")
elbow 
#somente para visualizar os valores do grafico podemos utilizar o comando
wss_values <- elbow$data
wss_values

#CALCULANDO SILHOUETE DE CADA PONTO
silhouette_values <- silhouette(grupos, d2)
#colocando nome nas linhas
rownames(silhouette_values) <- rownames(alunos_pap2)
#verificando os valores de cada ponto
silhouette_values
#grafico da silhueta
fviz_silhouette(silhouette_values, label = TRUE)

#VERIFICANDO GRAFICO SILHOUETTE para detectar sugestao de numero de grupos
fviz_nbclust(alunos_pap2, FUN = hcut, method = "silhouette", hc_method = "average")


# se por ventura quiser ver os valores dos pontos do grafico da silhouette, precisa pedir isso que vem da library(NbClust) 
silhouette_sugestao <- fviz_nbclust(alunos_pap2, FUN = hcut, method = "silhouette", hc_method = "average")
silhouette_sugestao$data
