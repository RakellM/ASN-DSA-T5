library(tidymodels)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(readxl)



celulas <- read_excel("dados/celulas.xlsx")
View(celulas)

celulas <- celulas %>%
  mutate(
    Classe = as.factor(Classe),
    Caudas = as.factor(Caudas),
    Nucleos = as.factor(Nucleos)
  )

celulas_tree_model <- decision_tree(min_n = 1, cost_complexity = -1, tree_depth=5) %>% 
  set_mode("classification") %>% 
  set_engine("rpart", parms = list(split = "gini")) #para entropia usar information

celulas_tree_fit <- fit(
  celulas_tree_model,
  Classe ~.,
  data = celulas
)

rpart.plot(celulas_tree_fit$fit, roundint=FALSE)


compra <- read_excel("dados/compra.xlsx")
View(compra)

compra <- compra %>%
  mutate(
    Compra = as.factor(Compra),
    Sexo = as.factor(Sexo),
    Pais = as.factor(Pais)
  )

compra_tree_model <- decision_tree(min_n = 1, cost_complexity = 0, tree_depth=5) %>% 
  set_mode("classification") %>% 
  set_engine("rpart", parms = list(split = "information")) #para gini usar gini

compra_tree_fit <- fit(
  compra_tree_model,
  Compra ~.,
  data = compra
)

rpart.plot(compra_tree_fit$fit, roundint=FALSE)




##########
# Exercicio
# faça uma árvore de decisao para os dados Seguro_Banco
# com min_n = 10, cost_complexity = 0.01, tree_depth=5
###########

Seguro_Banco <- read_excel("dados/Seguro_Banco.xlsx")
View(Seguro_Banco)

Seguro_Banco <- Seguro_Banco %>%
  mutate(
    Produto = as.factor(Produto),
    Residencia = as.factor(Residencia),
    Segmento = as.factor(Segmento)
  )

seguro_tree_model <- decision_tree(min_n = 10, cost_complexity = 0.01, tree_depth=5) %>% 
  set_mode("classification") %>% 
  set_engine("rpart", parms = list(split = "gini")) 

seguro_tree_fit <- fit(
  seguro_tree_model,
  Produto ~. -IDNUM,
  data = Seguro_Banco
)

rpart.plot(seguro_tree_fit$fit, roundint=FALSE)
