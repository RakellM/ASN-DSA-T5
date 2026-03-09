###########################
#
#  Pacotes necessarios
#
###########################

library(readxl)
library(corrplot)
library(REdaS)
library(psych)
library(tidyverse)

##1. importar os dados das notas dos alunos
dados <- read_excel("./ASN-DSA-T5/21_CP/Exemplo_Adriana/data/mao.xlsx")
View(dados)

##2. Matriz correlacao
MC <- cor(dados)
MC

# visualizar as correlacoes 
corrplot(MC)

##3. Estatistica KMO
# pela biblioteca REdaS
KMOS(dados)

# pela biblioteca Psych
KMO(dados)

##4. Teste de Bartlett
# H0: a matriz de correlacao e igual a matriz identidade
# H1: a matriz de correlacao e diferente a matriz identidade
cortest.bartlett(dados)

# Rejeita H0 

##5. Calculando os componentes
componentes_dados_sem_rotacao <- principal(dados, nfactors = 2, rotate = "none")
componentes_dados_sem_rotacao

##biplot
biplot(componentes_dados_sem_rotacao)

## autovalores
componentes_dados_sem_rotacao$values

## % da variabilidade
componentes_dados_sem_rotacao$Vaccounted
PC1 <- componentes_dados_sem_rotacao$Vaccounted[2,1]; PC1
PC2 <- componentes_dados_sem_rotacao$Vaccounted[2,2]; PC2

## scores fatoriais
componentes_dados_sem_rotacao$weights

## calculando Fatores na "mao"
dados_fatores <- dados %>% 
  mutate(ZPort= (Portugues - mean(Portugues)) / sd(Portugues),
         ZMat= (Matematica - mean(Matematica)) / sd(Matematica),
         ZFis= (Fisica - mean(Fisica)) / sd(Fisica)) %>% 
  mutate(F1 = componentes_dados_sem_rotacao$weights[1,1]*ZPort + 
           componentes_dados_sem_rotacao$weights[2,1]*ZMat+
           componentes_dados_sem_rotacao$weights[3,1]*ZFis,
         F2 = componentes_dados_sem_rotacao$weights[1,2]*ZPort + 
           componentes_dados_sem_rotacao$weights[2,2]*ZMat+
           componentes_dados_sem_rotacao$weights[3,2]*ZFis)

##scores
componentes_dados_sem_rotacao$scores
dados_fatores$scores_PC1 <- componentes_dados_sem_rotacao$scores[,1]
dados_fatores$scores_PC2 <- componentes_dados_sem_rotacao$scores[,2]

## cargas fatoriais (loadings)
componentes_dados_sem_rotacao$loadings



#### rotacionando
componentes_dados_com_rotacao <- principal(dados, nfactors = 2, rotate = "varimax")

#biplot
biplot(componentes_dados_com_rotacao)


#autovalores
componentes_dados_com_rotacao$values

# % da variabilidade
componentes_dados_com_rotacao$Vaccounted

# scores fatoriais
componentes_dados_com_rotacao$weights

## fatores rotacionados
dados_fatores_rotacao <- dados 
dados_fatores_rotacao$F1 <- componentes_dados_com_rotacao$scores[,1]
dados_fatores_rotacao$F2 <- componentes_dados_com_rotacao$scores[,2]


#cargas fatoriais (loading)
componentes_dados_com_rotacao$loadings


## ranking
componentes_dados_com_rotacao$Vaccounted
PC1_r <- componentes_dados_com_rotacao$Vaccounted[2,1]; PC1_r
PC2_r <- componentes_dados_com_rotacao$Vaccounted[2,2]; PC2_r
dados_fatores_rotacao <- dados_fatores_rotacao %>% 
  mutate(ranking= (PC1_r*F1)+ (PC2_r*F2)) %>% 
  arrange(desc(ranking))

View(dados_fatores_rotacao)


