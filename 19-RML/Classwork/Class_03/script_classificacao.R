# ==================================================
# 1. Carregar pacotes
# ==================================================

# Instala pacotes se ainda não estiverem instalados
install.packages("tidymodels", dependencies = TRUE)
install.packages("tidyverse", dependencies = TRUE)
install.packages("ranger", dependencies = TRUE)
install.packages("xgboost", dependencies = TRUE)
install.packages("dlookr", dependencies = TRUE)
install.packages("corrplot", dependencies = TRUE)
install.packages("kernlab", dependencies = TRUE)

# Carregando pacotes
library(tidymodels)   # traz recipes, parsnip, workflows, tune, yardstick, dials, rsample
library(tidyverse)    # manipulação e visualização
library(ranger)       # Random Forest
library(xgboost)      # XGBoost
library(dlookr)       # estatísticas descritivas
library(corrplot)     # gráfico de correlação
library(kernlab)      # Engine para SVM


# ==================================================
# 2. Importar e Preparar a base de dados
# ==================================================

df = read_csv("aula_02_exemplo_02.csv")
df$custo_medico = factor(df$custo_medico,levels = c("Alto_Custo", "Baixo_Custo"))


# Visualizar as primeiras linhas
head(df)
nrow(df)

# ==================================================
# 4. Análise Exploratória
# ==================================================
diagnose(df)
diagnose_numeric(df)
diagnose_category(df)

# Gráfico de barras para a nova variável alvo
df %>%
  ggplot(aes(x = custo_medico, fill = custo_medico)) +
  geom_bar() +
  labs(title = "Distribuição de Custo Médico", x = "Categoria de Custo", y = "Contagem")

# Relação entre ser fumante e o custo médico
df %>%
  ggplot(aes(x = smoker, fill = custo_medico)) +
  geom_bar(position = "fill") +
  labs(title = "Proporção de Custo Médico por Hábito de Fumar",
       x = "Fumante", y = "Proporção")


# ==================================================
# 5. Correlações numéricas
# ==================================================
correlacao = df %>% 
  select(age, bmi, children) %>%
  cor(method = "spearman")

corrplot(correlacao, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45)


# ==================================================
# 6. Divisão dos Dados
# ==================================================

# Dividir 80% treino / 20% teste, estratificando pela variável alvo
set.seed(123)
split = initial_split(df, prop = 0.8, strata = custo_medico)
train_data = training(split)
test_data  = testing(split)

# ==================================================
# 7. Pré-processamento e Transformação dos dados (Receita)
# ==================================================

# A receita agora prevê `custo_medico`
rec = recipe(custo_medico ~ ., data = train_data) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ==================================================
# 8. Definição dos Modelos e Tunagem de Hiperparâmetros
# ==================================================

# --- Definição dos Modelos de CLASSIFICAÇÃO ---

# 1. Regressão Logística (com regularização)
log_model = logistic_reg(penalty = tune(), mixture = 1) %>% # mixture = 1 é Lasso (L1)
  set_engine("glmnet") %>%
  set_mode("classification")

# 2. Random Forest
rf_model = rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# 3. XGBoost
xgb_model = boost_tree(
  trees = 1000,
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# --- Workflows: Combina receita e modelo ---
log_wf = workflow() %>% add_recipe(rec) %>% add_model(log_model)
rf_wf  = workflow() %>% add_recipe(rec) %>% add_model(rf_model)
xgb_wf = workflow() %>% add_recipe(rec) %>% add_model(xgb_model)
# --- Validação Cruzada ---
set.seed(123)
cv_folds = vfold_cv(train_data, v = 5, strata = custo_medico)

# --- Grids para Tuning ---
set.seed(123)
log_grid = grid_random(penalty(), size = 10)
rf_grid = grid_random(mtry(range = c(1, 8)), min_n(range = c(2, 10)), size = 10)
xgb_grid = grid_random(tree_depth(range = c(2, 10)), learn_rate(range = c(0.01, 0.3)), loss_reduction(), size = 10)

# --- Métricas de Classificação ---
class_metrics = metric_set(roc_auc, precision,recall)

# --- Ajuste dos modelos com validação cruzada (Tuning) ---
log_tune = tune_grid(log_wf, resamples = cv_folds, grid = log_grid, metrics = class_metrics)
rf_tune  = tune_grid(rf_wf, resamples = cv_folds, grid = rf_grid, metrics = class_metrics)
xgb_tune = tune_grid(xgb_wf, resamples = cv_folds, grid = xgb_grid, metrics = class_metrics)

# --- Seleção dos melhores hiperparâmetros (baseado em roc_auc) ---
best_log = select_best(log_tune, metric = "roc_auc")
best_rf  = select_best(rf_tune, metric = "roc_auc")
best_xgb = select_best(xgb_tune, metric = "roc_auc")

# --- Finaliza workflows com melhores parâmetros ---
final_log_wf = finalize_workflow(log_wf, best_log)
final_rf_wf  = finalize_workflow(rf_wf, best_rf)
final_xgb_wf = finalize_workflow(xgb_wf, best_xgb)

# --- Ajusta os modelos finais nos dados de treino ---
final_log_fit = fit(final_log_wf, data = train_data)
final_rf_fit  = fit(final_rf_wf, data = train_data)
final_xgb_fit = fit(final_xgb_wf, data = train_data)

# ==================================================
# 9. Previsões Conjunto de Treino e Teste
# ==================================================

# Previsões - Regressão Logística (Treino)
pred_log_treino =predict(final_log_fit, new_data = train_data, type = "prob") %>%
  bind_cols(train_data)

# Previsões - Random Forest (Treino)
pred_rf_treino = predict(final_rf_fit, new_data = train_data, type = "prob") %>%
  bind_cols(train_data)

# Previsões - XGBoost (Treino)
pred_xgb_treino = predict(final_xgb_fit, new_data = train_data,type = "prob") %>%
  bind_cols(train_data)



# --- Previsões no conjunto de TESTE ---

# Previsões - Regressão Logística (Teste)
pred_log_teste = predict(final_log_fit, new_data = test_data,type = "prob") %>%
  bind_cols(test_data)

# Previsões - Random Forest (Teste)
pred_rf_teste = predict(final_rf_fit, new_data = test_data,type = "prob") %>%
  bind_cols(test_data)

# Previsões - XGBoost (Teste)
pred_xgb_teste = predict(final_xgb_fit, new_data = test_data,type = "prob") %>%
  bind_cols(test_data)


# ==================================================
# 10. Analise de limiar com os dados de treino
# ==================================================


# --- Aplicando limiares aos dataframes de TREINO ---


# Regressão Logística - Treino
pred_log_treino = pred_log_treino %>%
  mutate(
    .pred_class_0.1 = factor(ifelse(.pred_Alto_Custo >= 0.1, "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.5 = factor(ifelse(.pred_Alto_Custo >= 0.5,  "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.7 = factor(ifelse(.pred_Alto_Custo >= 0.7,  "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.9 = factor(ifelse(.pred_Alto_Custo >= 0.9,  "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo"))
  )

# Random Forest - Treino
pred_rf_treino = pred_rf_treino %>%
  mutate(
    .pred_class_0.1 = factor(ifelse(.pred_Alto_Custo >= 0.1, "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.5 = factor(ifelse(.pred_Alto_Custo >= 0.5,  "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.7 = factor(ifelse(.pred_Alto_Custo >= 0.7,  "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.9 = factor(ifelse(.pred_Alto_Custo >= 0.9, "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo"))
  )

# XGBoost - Treino
pred_xgb_treino = pred_xgb_treino %>%
  mutate(
    .pred_class_0.1 = factor(ifelse(.pred_Alto_Custo >= 0.1,  "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.5 = factor(ifelse(.pred_Alto_Custo >= 0.5,  "Alto_Custo", "Baixo_Custo"),levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.7 = factor(ifelse(.pred_Alto_Custo >= 0.7,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.9 = factor(ifelse(.pred_Alto_Custo >= 0.9,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo"))
  )

# --- Aplicando limiares aos dataframes de TESTE ---

# Regressão Logística - Teste
pred_log_teste = pred_log_teste %>%
  mutate(
    .pred_class_0.1 = factor(ifelse(.pred_Alto_Custo >= 0.1,  "Alto_Custo", "Baixo_Custo"), levels =  c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.5 = factor(ifelse(.pred_Alto_Custo >= 0.5,  "Alto_Custo", "Baixo_Custo"), levels =  c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.7 = factor(ifelse(.pred_Alto_Custo >= 0.7,  "Alto_Custo", "Baixo_Custo"), levels =  c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.9 = factor(ifelse(.pred_Alto_Custo >= 0.9,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo"))
  )

# Random Forest - Teste
pred_rf_teste = pred_rf_teste %>%
  mutate(
    .pred_class_0.1 = factor(ifelse(.pred_Alto_Custo >= 0.1,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.5 = factor(ifelse(.pred_Alto_Custo >= 0.5,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.7 = factor(ifelse(.pred_Alto_Custo >= 0.7,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.9 = factor(ifelse(.pred_Alto_Custo >= 0.9,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo"))
  )

# XGBoost - Teste
pred_xgb_teste = pred_xgb_teste %>%
  mutate(
    .pred_class_0.1 = factor(ifelse(.pred_Alto_Custo >= 0.1,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.5 = factor(ifelse(.pred_Alto_Custo >= 0.5,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.7 = factor(ifelse(.pred_Alto_Custo >= 0.7,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo")),
    .pred_class_0.9 = factor(ifelse(.pred_Alto_Custo >= 0.9,  "Alto_Custo", "Baixo_Custo"), levels = c("Alto_Custo", "Baixo_Custo"))
  )


# ==================================================
# 11. Cálculo de Métricas para Cada Limiar (APENAS PARA OS DADOS DE TREINO)
# ==================================================

metricas_utilizadas = metric_set(precision, recall,f_meas)

metricas_log_treino = bind_rows(
  pred_log_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.1,event_level = "first") %>% mutate(limiar = 0.1),
  pred_log_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(limiar = 0.5),
  pred_log_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.7,event_level = "first") %>% mutate(limiar = 0.7),
  pred_log_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.9,event_level = "first") %>% mutate(limiar = 0.9)
) %>% mutate(Modelo = "Regressão Logística")


metricas_rf_treino = bind_rows(
  pred_rf_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.1,event_level = "first") %>% mutate(limiar = 0.1),
  pred_rf_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(limiar = 0.5),
  pred_rf_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.7,event_level = "first") %>% mutate(limiar = 0.7),
  pred_rf_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.9,event_level = "first") %>% mutate(limiar = 0.9)
) %>% mutate(Modelo = "Random Forest")

metricas_xgb_treino = bind_rows(
  pred_xgb_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.1,event_level = "first") %>% mutate(limiar = 0.1),
  pred_xgb_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(limiar = 0.5),
  pred_xgb_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.7,event_level = "first") %>% mutate(limiar = 0.7),
  pred_xgb_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.9,event_level = "first") %>% mutate(limiar = 0.9)
) %>% mutate(Modelo = "XGBoost")


metricas_log_treino %>% ggplot(aes(x = limiar, y = .estimate , color = .metric,group = .metric)) +
  geom_line(linewidth = 1.2) + # Adiciona as linhas
  geom_point(size = 3)        # Adiciona pontos para destacar os limiares


metricas_rf_treino %>% ggplot(aes(x = limiar, y = .estimate , color = .metric,group = .metric)) +
  geom_line(linewidth = 1.2) + # Adiciona as linhas
  geom_point(size = 3)        # Adiciona pontos para destacar os limiares


metricas_xgb_treino %>% ggplot(aes(x = limiar, y = .estimate , color = .metric,group = .metric)) +
  geom_line(linewidth = 1.2) + # Adiciona as linhas
  geom_point(size = 3)        # Adiciona pontos para destacar os limiares



# ==================================================
# 11.5. Análise Visual de Performance (Dados de Treino)
# ==================================================

matriz_conf_log = pred_log_treino %>%
  conf_mat(truth = custo_medico, estimate = .pred_class_0.5)

autoplot(matriz_conf_log, type = "heatmap") +
  labs(title = "Matriz de Confusão - Regressão Logística (Treino, Limiar 0.5)") +
  scale_fill_gradient(low = "#D6EAF8", high = "#2E86C1")

matriz_conf_rf = pred_rf_treino %>%
  conf_mat(truth = custo_medico, estimate = .pred_class_0.5)

autoplot(matriz_conf_rf, type = "heatmap") +
  labs(title = "Matriz de Confusão - Random Forest (Treino, Limiar 0.5)") +
  scale_fill_gradient(low = "#D6EAF8", high = "#2E86C1")


matriz_conf_xgboost = pred_xgb_treino %>%
  conf_mat(truth = custo_medico, estimate = .pred_class_0.5)

autoplot(matriz_conf_xgboost, type = "heatmap") +
  labs(title = "Matriz de Confusão - XGBOOST (Treino, Limiar 0.5)") +
  scale_fill_gradient(low = "#D6EAF8", high = "#2E86C1")



# ==================================================
# 12. Consolidação e Análise dos Resultados
# ==================================================

# ==================================================
# Calcula métricas para dados de TREINO
# ==================================================
metricas_log_treino = pred_log_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(Modelo = "Regressão Logística")
metricas_rf_treino  = pred_rf_treino  %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(Modelo = "Random Forest")
metricas_xgb_treino = pred_xgb_treino %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(Modelo = "XGBoost")

resultados_treino = bind_rows(
  metricas_log_treino,
  metricas_rf_treino,
  metricas_xgb_treino
)

# ==================================================
# Calcula métricas para dados de TESTE
# ==================================================

metricas_log_teste = pred_log_teste %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(Modelo = "Regressão Logística")
metricas_rf_teste  = pred_rf_teste  %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(Modelo = "Random Forest")
metricas_xgb_teste = pred_xgb_teste %>% metricas_utilizadas(truth = custo_medico, estimate = .pred_class_0.5,event_level = "first") %>% mutate(Modelo = "XGBoost")

resultados_teste = bind_rows(
  metricas_log_teste,
  metricas_rf_teste,
  metricas_xgb_teste
)

# ==================================================
# Junta e Apresenta Resultados Finais
# ==================================================

# Adiciona coluna indicando etapa
resultados_teste$etapa = "Teste"
resultados_treino$etapa = "Treino"

View(resultados_treino %>% bind_rows(resultados_teste) %>% filter(.metric == "f_meas") %>% arrange(desc(Modelo), desc(etapa) ))


# ==================================================
# 13. Treinar o modelo final e simular produção
# ==================================================

# A regressão logística foi o melhor. Vamos treiná-lo com todos os dados.
modelo_final = fit(final_log_wf, data = df)

# Salvando o modelo
saveRDS(modelo_final, "modelo_final_classificacao.rds")

# --- Simulação de Produção ---
cat("\n\n--- Simulação de Produção ---\n")

# Carregando o modelo salvo
workflow_prod = readRDS("modelo_final_classificacao.rds")

dados_prod = read.csv("dados_producao_classificacao.csv")

# função para automatizar a predição
realizar_predicao = function(workflow, dados){

  previsoes = predict(workflow, new_data = dados, type = "prob") %>%
  bind_cols(dados)
  
  return(previsoes)
}

# Previsões diretas
previsoes = realizar_predicao(workflow_prod,dados_prod)
