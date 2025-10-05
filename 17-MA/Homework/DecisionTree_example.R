
# Path to the main project directory
setwd("C:/Users/kel_m/OneDrive/Project_Code/ASN-DSA-T5/17-MA")

# Load libraries
library(tidymodels)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(readxl)


# Install packages if not already installed
# install.packages("rpart.plot")
# install.packages("tidymodels")


# EXERCISE 1
# Load data
celulas <- read_excel("data/celulas.xlsx")
View(celulas)

# change variables to factors
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


# EXERCISE 2
# Load data
compra <- read_excel("data/compra.xlsx")
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
