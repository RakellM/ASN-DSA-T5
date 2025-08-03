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
library(readxl)

########################################
#
#   CLUSTER HIERARQUICO 
#
########################################

#LEITURA DOS DADOS
alunos_pap <- read.table("dados/alunos_pap.csv", sep = ";", header = T, dec = ",")
View(alunos_pap)
rownames(alunos_pap) <- alunos_pap[,1]
alunos_pap <- alunos_pap[,-1]

#CALCULANDO MATRIZ DE DISTANCIAS
d <- dist(alunos_pap, method = "euclidean")
d

#DEFININDO O CLUSTER A PARTIR DO METODO ESCOLHIDO
#metodos disponiveis "average", "single", "complete" e "ward.D"
hc1 <- hclust(d, method = "single" )
hc2 <- hclust(d, method = "complete" )
hc3 <- hclust(d, method = "average" )
hc4 <- hclust(d, method = "ward.D" )

#DESENHANDO O DENDOGRAMA
plot(hc1, cex = 0.6, hang = -1)
plot(hc2, cex = 0.6, hang = -1)
plot(hc3, cex = 0.6, hang = -1)
plot(hc4, cex = 0.9, hang = -1)

#BRINCANDO COM O DENDOGRAMA PARA 2 GRUPOS
rect.hclust(hc4, k = 2)

#CURIOSIDADE
#COMPARANDO DENDOGRAMAS
#comparando o metodo average com ward
dend3 <- as.dendrogram(hc3)
dend4 <- as.dendrogram(hc4)
dend_list <- dendlist(dend3, dend4) 
#EMARANHADO, quanto menor, mais iguais os dendogramas sao
tanglegram(dend3, dend4, main = paste("Emaranhado =", round(entanglement(dend_list),2)))
#agora comparando o metodo single com complete
dend1 <- as.dendrogram(hc1)
dend2 <- as.dendrogram(hc2)
dend_list2 <- dendlist(dend1, dend2) 
#EMARANHADO, quanto menor, mais iguais os dendogramas sao
tanglegram(dend1, dend2, main = paste("Emaranhado =", round(entanglement(dend_list2),2)))

#criando 2 grupos de alunos
grupo_alunos2 <- cutree(hc4, k = 2)
table(grupo_alunos2)

#transformando em data frame a saida do cluster
alunos_grupos <- data.frame(grupo_alunos2)

#juntando com a base original
Base_alunos_fim <- cbind(alunos_pap, alunos_grupos)

#entendendo os clusters
#FAZENDO ANALISE DESCRITIVA
#MEDIAS das variaveis por grupo (tambem eh interessante ver desvio padrao)
mediagrupo_alunos <- Base_alunos_fim %>% 
  group_by(grupo_alunos2) %>% 
  summarise(n = n(),
            Portugues = mean(Portugues), 
            Matematica = mean(Matematica))
mediagrupo_alunos

# apos os grupos criados, vale fazer algumas analises de comparacao de medias
# como essa base eh muito pequena, nao farei nenhum teste
# mas faremos em outro exemplo!
