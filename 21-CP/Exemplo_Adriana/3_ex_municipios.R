########################################
#
#   CHAMANDO BIBLIOTECAS 
#
########################################

library(tidyverse)
library(corrplot)
library(REdaS)
library(psych)
library(FactoMineR)
library(factoextra)


#carregar base municipio
municipios <- read_rds("./ASN-DSA-T5/21_CP/Exemplo_Adriana/data/municipios.rds")

#matriz de correlacao
correlacao <- cor(municipios[,2:7])
print(correlacao, digits = 2)

# visualizar as correlacoes 
corrplot(correlacao)

#KMO
KMOS(municipios[,2:7])

# teste de Bartlett
cortest.bartlett(municipios[,2:7])


#pca com a matriz de correlacao
mun_pca <- PCA(municipios[,2:7], scale.unit = T, graph = T, ncp=7)
summary(mun_pca)
fviz_screeplot(mun_pca)
windows(); fviz_screeplot(mun_pca)

# analise de componentes principais
PCAmun<-principal(municipios[,2:7], nfactors = 2, rotate = "none", scores = TRUE)
PCAmun

## biplot(sem rotacao) 
biplot(PCAmun) 
windows(); biplot(PCAmun) 

#cargas fatoriais
PCAmun$loadings

#rotacionando
PCAmunvarimax<-principal(municipios[,2:7], nfactors = 2, rotate = "varimax", scores = TRUE)
PCAmunvarimax

## biplot(com rotacao) 
windows(); biplot(PCAmunvarimax) 

#quanto cada variavel esta relacionada com o fator - carga fatorial
PCAmunvarimax$loadings


#dados dos fatores
municipios$scores_PCA1 <- PCAmunvarimax$scores[,1]
municipios$scores_PCA2 <- PCAmunvarimax$scores[,2]


## criando ranking
PCAmunvarimax$Vaccounted
PC1 <- PCAmunvarimax$Vaccounted[2,1]; PC1
PC2 <- PCAmunvarimax$Vaccounted[2,2]; PC2

municipios <- municipios %>% 
  mutate(ranking = ((PC1)*scores_PCA1) + ((PC2)*scores_PCA2)) %>% 
  arrange(desc(ranking))

View(municipios)

