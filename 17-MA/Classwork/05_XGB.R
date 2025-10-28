# Gradient Boosting
# pacotes
library(tidymodels)
library(tidyverse)
library(readxl)
library(xgboost)


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

# Aqui estamos soh definindo a equacao e fazendo o tratamento nos dados
ABT_receita <- recipe(Produto ~ ., data = ABT_treino) %>% 
  step_rm(IDNUM) %>% 
  step_corr(all_numeric(), threshold = 0.75) %>% # tirando todas as variaveis que são correlacionadas entre si acima de 0,75
# step_unknown(all_nominal(), -Produto) %>% # criando unknow se tiver missing em alguma categorica, coisa que nao tem
  step_zv(all_predictors()) %>% #removendo variaveis sem variabilidade 
  step_dummy(all_nominal(), -Produto) %>% # criando dummie de todas as categoricas
  step_impute_mean(all_numeric()) #adicionando a media nas numericas com dados faltantes
# step_normalize(all_numeric()) %>%
# step_modeimpute(all_nominal(), -Produto) %>%
# step_bagimpute(all_numeric(), trees = 1)
# existem varias opcoes, vai depender de caso a caso, entao vale consultar
# https://recipes.tidymodels.org/reference/index.html 

#encapsulando a receita
Receita <- prep(ABT_receita)
Receita

#visualisando a ABT tratada pela Receita
ABT_treino_tratada <- juice(Receita) 
ABT_treino_tratada

# se quiser colocar na base de teste, pode fazer isso, apenas para visualizacao
# pois isso eh feito la em baixo
ABT_teste_tratada <- bake(Receita, new_data = ABT_teste) 


# define o modelo (definimos os hiperparamentos ou se vamos procurar 
# por eles no crossvalidation) 
Seguro_model <- boost_tree(
  mtry = 4, 
  min_n = 30, 
  tree_depth = 4, 
  trees = 100, 
  sample_size = 0.70, 
  learn_rate = 0.3, 
  loss_reduction = tune()) %>%
  set_mode("classification") %>%
  set_engine("xgboost", lambda = 0)

# cria o fluxo logico do modelo (workflow)
Seguro_workflow <- workflow() %>% 
  add_recipe(ABT_receita) %>% 
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

# verificando as medidas 
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
  accuracy(truth = Produto, estimate = minha_predicao)


# modelo final, sendo recalculado pra base toda, com os melhores hiper
ABT_modelo_final <- Seguro_workflow %>% fit(data = ABT)

