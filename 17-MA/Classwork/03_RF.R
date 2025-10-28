#Random Forest
# pacotes
library(tidymodels)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(readxl)


#trazendo dados do Seguro
Seguro_Banco <- read_excel("dados/Seguro_Banco.xlsx")
View(Seguro_Banco)

# transformando a variavel resposta em factor
ABT <- Seguro_Banco %>%
  mutate(Produto = factor(ifelse(Produto == 0, "Nao", "Sim")))

# Particionando os dados (treino e teste) 
set.seed(1987)
divisao_treino_teste <- initial_split(ABT, strata = "Produto", prop = 0.75)

ABT_treino <- training(divisao_treino_teste)
ABT_teste  <- testing(divisao_treino_teste)

mean(ABT$Produto== "Sim")
mean(ABT_treino$Produto== "Sim")
mean(ABT_teste$Produto== "Sim")

# Aqui estamos soh definindo a equacao, mas ainda vamos usar melhor essa funcao
ABT_tratamento <- recipe(Produto ~ ., data = ABT_treino) %>% 
  step_rm(IDNUM)

# define o modelo (definimos os hiperparamentos ou se vamos procurar 
# por eles no crossvalidation) 
Seguro_model <- rand_forest(
  min_n = tune(),
  trees = tune(),
  mtry = tune()
) %>% 
  set_engine("ranger") %>%
  set_mode("classification") 


# cria o fluxo logico do modelo (workflow)
Seguro_workflow <- workflow() %>% 
  add_recipe(ABT_tratamento) %>% 
  add_model(Seguro_model) 

# define o seu número de folds do método de crossvalidation
Seguro_folds <- vfold_cv(ABT_treino, v = 5)

# Para fazer seus testes, você pode estipular o numero de parametros 
# que quer testar para cada
# tune que marcou no escopo do modelo
Num_hiperparam <- 2


# Agora sim vamos "tunar" o modelo
Tunando_Modelo <- tune_grid(
  Seguro_workflow, 
  resamples = Seguro_folds,
  grid = Num_hiperparam,
  metrics = metric_set(accuracy, precision, recall)
)

# inspecao da tunagem 
metricas <- collect_metrics(Tunando_Modelo)
metricas
show_best(Tunando_Modelo, "accuracy")


# seleciona o melhor conjunto de hiperparametros
ABT_melhores_hiper <- select_best(Tunando_Modelo, "accuracy")
Seguro_workflow <- Seguro_workflow %>% finalize_workflow(ABT_melhores_hiper)


# vendo desempenho do modelo na base de teste
Seguro_workflow_fit <- Seguro_workflow %>% last_fit(split = divisao_treino_teste)

collect_metrics(Seguro_workflow_fit)
predicoes <- collect_predictions(Seguro_workflow_fit)
predicoes

predicoes %>% 
  mutate(
    minha_predicao = factor(ifelse(.pred_Sim > 0.55, "Sim", "Nao"))
  ) %>%
 # accuracy(truth = Produto, estimate = minha_predicao)
  conf_mat(Produto, minha_predicao)


# modelo final, sendo recalculado pra base toda, com os melhores hiper
ABT_modelo_final <- Seguro_workflow %>% fit(data = ABT)


