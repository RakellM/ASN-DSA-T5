########################################
#
#   CHAMANDO BIBLIOTECAS 
#
########################################

library(tidyverse)
library(REdaS)
library(psych)
library(FactoMineR)
library(factoextra)

#Carregar base de dados: 
mcdonalds <- read_rds("./ASN-DSA-T5/21_CP/Exemplo_Adriana/data/mcdonald.rds")

## calculando KMO a partir da biblioteca REdaS
KMOS(mcdonalds[,2:12])
KMOS(mcdonalds[,3:12])
KMOS(mcdonalds[,4:12])

## calculando Teste de Bartlett pela biblioteca psych
cortest.bartlett(mcdonalds[,4:12])

## calculando os componentes
componentes_sem_rotacao <- principal(mcdonalds[,4:12], nfactors=2, rotate="none", scores=T)
componentes_sem_rotacao

## biplot(semrotacao) 
biplot(componentes_sem_rotacao) 

# tem como ver o grafico de outra forma, quem sabe mais visual?
# library FactoMineR
componentes_sem_rotacao_1 <- PCA(mcdonalds[,4:12], scale.unit = T, graph = T, ncp=2)
summary(componentes_sem_rotacao_1)

# consigo ver isso de forma visual? o percentual de cada componente?
# library factoextra
fviz_screeplot(componentes_sem_rotacao_1)

## autovalores
componentes_sem_rotacao$values

## % da variabilidade
componentes_sem_rotacao$Vaccounted
PC1 <- componentes_sem_rotacao$Vaccounted[2,1]; PC1
PC2 <- componentes_sem_rotacao$Vaccounted[2,2]; PC2

## scores fatoriais
componentes_sem_rotacao$weights

## cargas fatoriais (loadings no R)
componentes_sem_rotacao$loadings


# calculando fatores
# extraindo scores 
mcdonalds_sem_rotacao <- mcdonalds
mcdonalds_sem_rotacao$scores_PCA1 <- componentes_sem_rotacao$scores[,1]
mcdonalds_sem_rotacao$scores_PCA2 <- componentes_sem_rotacao$scores[,2]





## rotacionando
componentes_com_rotacao <- principal(mcdonalds[,4:12], nfactors=2, rotate="varimax", scores=T)

## biplot(comrotacao) 
biplot(componentes_com_rotacao) 

## autovalores
componentes_com_rotacao$values

## % da variabilidade
componentes_com_rotacao$Vaccounted
PC1 <- componentes_com_rotacao$Vaccounted[2,1];PC1
PC2 <- componentes_com_rotacao$Vaccounted[2,2];PC2

## scores fatoriais
componentes_com_rotacao$weights

## cargas fatoriais (loadings no R)
componentes_com_rotacao$loadings


# calculando fatores
# ou seja, mesma coisa que pegar direto do objeto 
mcdonalds_com_rotacao <- mcdonalds
mcdonalds_com_rotacao$scores_PCA1 <- componentes_com_rotacao$scores[,1]
mcdonalds_com_rotacao$scores_PCA2 <- componentes_com_rotacao$scores[,2]

## criando ranking
mcdonalds_com_rotacao <- mcdonalds_com_rotacao %>% 
  mutate(ranking = (PC1*scores_PCA1) + (PC2*scores_PCA2)) %>% 
  arrange(desc(ranking))
