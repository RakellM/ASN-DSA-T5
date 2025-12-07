# pacotes
library(tidymodels)
library(tidyverse)
library(readxl)
library(glmnet)


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

#pequena exploratoria dos dados
skimr::skim(ABT_treino)
visdat::vis_miss(ABT_treino)
ABT_treino %>%
  select(where(is.numeric)) %>%
  cor(use = "pairwise.complete.obs") %>%
  corrplot::corrplot()

# Aqui estamos soh definindo a equacao, mas ainda vamos usar melhor essa funcao
ABT_receita <- recipe(Produto ~ ., data = ABT_treino) %>% 
  step_rm(IDNUM) %>% 
  step_corr(all_numeric(), threshold = 0.75) %>% # tirando todas as variaveis que são correlacionadas entre si acima de 0,90
  step_zv(all_predictors()) %>% #removendo variaveis sem variabilidade 
  step_dummy(all_nominal(), -Produto) %>% # criando dummie de todas as categoricas
  step_impute_mean(all_numeric()) %>%  #adicionando a media nas numericas com dados faltantes
  step_normalize(all_numeric()) #normalizando para o Lasso
# step_modeimpute(all_nominal(), -Produto)
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
Seguro_model <- logistic_reg(penalty = tune(), 
                             mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# cria o fluxo logico do modelo (workflow)
Seguro_workflow <- workflow() %>% 
  add_recipe(ABT_receita) %>% 
  add_model(Seguro_model) 

# define o seu número de folds do método de crossvalidation
Seguro_folds <- vfold_cv(ABT_treino, v = 5)

# Para fazer seus testes, você pode estipular o numero de parametros 
# que quer testar para cada
# tune que marcou no escopo do modelo
# neste caso estou pedindo para procurar nesse range
Num_hiperparam <-10


# Agora sim vamos "tunar" o modelo
Tunando_Modelo <- tune_grid(
  Seguro_workflow, 
  resamples = Seguro_folds,
  grid = Num_hiperparam,
  metrics = metric_set(
    mn_log_loss, 
    accuracy,
    roc_auc)
  )


# verificando as medidas 
metricas <- collect_metrics(Tunando_Modelo)
metricas
autoplot(Tunando_Modelo)
show_best(Tunando_Modelo, "mn_log_loss")

# seleciona o melhor conjunto de hiperparametros
ABT_melhores_hiper <- select_best(Tunando_Modelo, "mn_log_loss")
Seguro_workflow <- Seguro_workflow %>% finalize_workflow(ABT_melhores_hiper)


# vendo desempenho do modelo na base de teste
Seguro_workflow_fit <- Seguro_workflow %>% last_fit(split = divisao_treino_teste)

collect_metrics(Seguro_workflow_fit)

vip::vi(Seguro_workflow_fit$.workflow[[1]]$fit$fit)


# modelo final, sendo recalculado pra base toda, com os melhores hiper
ABT_modelo_final <- Seguro_workflow %>% fit(data = ABT)

# importancia das variaveis
vip::vi(ABT_modelo_final$fit$fit)

#visualizando
vip::vip(ABT_modelo_final$fit$fit)




#Se fosse Regressao Linear o que mudaria?

#nao faria amostra estratificada
#divisao_treino_teste <- initial_split(ABT, strata = "Produto", prop = 0.75)

#carcaca do modelo
# Seguro_model <- linear_reg(penalty = tune(), mixture = 1) %>%
#   set_mode("regression") %>%
#   set_engine("glmnet")
# 
# #pedindo medidas adequadas
# Tunando_Modelo <- tune_grid(
#   Seguro_workflow, 
#   resamples = Seguro_folds,
#   grid = Num_hiperparam,
#   metrics = metric_set(rmse, mae)
# )
#e tambem:
#ABT_melhores_hiper <- select_best(Tunando_Modelo, "rmse")