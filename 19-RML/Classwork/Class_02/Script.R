# ==================================================
# 1. Carregar pacotes
# ==================================================

install.packages("tidymodels", dependencies = TRUE)
install.packages("tidyverse", dependencies = TRUE)
install.packages("ranger", dependencies = TRUE)
install.packages("xgboost", dependencies = TRUE)
install.packages("dlookr", dependencies = TRUE)
install.packages("corrplot", dependencies = TRUE)

# Carregando pacotes
library(tidymodels)   # traz recipes, parsnip, workflows, tune, yardstick, dials, rsample
library(tidyverse)    # manipulação e visualização
library(ranger)       # Random Forest
library(xgboost)      # XGBoost
library(dlookr)       # estatísticas descritivas
library(corrplot)     # gráfico de correlação



# ==================================================
# 2. Importar a base de dados
# ==================================================

df = read_csv("aula_01_exemplo_01.csv")

# Visualizar as primeiras linhas
head(df)

nrow(df)

# ==================================================
# 3. Estatísticas descritivas
# ==================================================
diagnose(df)

# Estatísticas numéricas
diagnose_numeric(df)

# Estatísticas categóricas
diagnose_category(df)

# ==================================================
# 4. Visualizações
# ==================================================

# Histograma das variáveis numéricas
plot_hist_numeric(df)

# Boxplot do IMC por sexo
df %>%
  ggplot(aes(x = sex, y = bmi, fill = sex)) +  # Define eixo x como sexo, eixo y como BMI, cor preenchimento por sexo
  geom_boxplot() +                             # Plota boxplot mostrando mediana, quartis e outliers
  labs(title = "Distribuição do IMC por Sexo", x = "Sexo", y = "IMC")  # Adiciona título e rótulos aos eixos

# Boxplot dos charges por fumante
df %>%
  ggplot(aes(x = smoker, y = charges, fill = smoker)) +  # Define eixo x como fumante, eixo y como charges, cor por fumante
  geom_boxplot() +                                       # Plota boxplot
  labs(title = "Custos Médicos por Hábito de Fumar", x = "Fumante", y = "Charges")  # Título e rótulos

# Boxplot dos charges por sexo
df %>%
  ggplot(aes(x = sex, y = charges, fill = sex)) +  # Define eixo x como sexo, eixo y como charges, cor por sexo
  geom_boxplot() +                                # Plota boxplot
  labs(title = "Custos Médicos por Sexo", x = "Sexo", y = "Charges")  # Título e rótulos

# Relação entre BMI e Charges
df %>%
  ggplot(aes(x = bmi, y = charges, color = smoker)) +  # Define eixo x como BMI, y como charges, cor por fumante
  geom_point(alpha = 0.7) +                            # Plota pontos, com transparência 0.7
  labs(title = "Relação entre IMC e Custos Médicos", x = "IMC", y = "Charges")  # Título e rótulos

# Média de charges por região
df %>%
  group_by(region) %>%                                           # Agrupa os dados por região
  summarise(media_charges = mean(charges, na.rm = TRUE)) %>%    # Calcula a média de charges por região
  ggplot(aes(x = region, y = media_charges, fill = region)) +   # Define eixo x como região, y como média de charges, cor por região
  geom_col() +                                                   # Plota gráfico de colunas
  labs(title = "Média de Custos por Região", x = "Região", y = "Charges Médios")  # Título e rótulos


# age  vs bmi

df %>%
  ggplot(aes(x = age, y = bmi)) +
  geom_point(alpha = 0.7) +
  labs(title = "Relação entre age e BMI", x = "age", y = "bmi")

# age  vs children
df %>%
  ggplot(aes(x = age, y = children)) +
  geom_point(alpha = 0.7) +
  labs(title = "Relação entre age e children", x = "age", y = "children")

# age  vs charges
df %>%
  ggplot(aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.7) +
  labs(title = "Relação entre age e charges", x = "age", y = "charges")


# ==================================================
# 5. Correlações numéricas
# ==================================================
correlacao = df %>%
  select(age, bmi, children, charges) %>%
  cor(method = "spearman")

corrplot(correlacao, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45)


# ==================================================
# 6. Divisão dos Dados
# ==================================================

# Dividir 80% treino / 20% teste

set.seed(123)
split = initial_split(df, prop = 0.8)
train_data = training(split)
test_data  = testing(split)

# ==================================================
# 7. Imputação de Transformação dos dados
# ==================================================

#criar a receita
rec = recipe(charges ~ ., data = train_data) %>%
  # Imputar valores faltantes pela mediana
  step_impute_median(all_numeric_predictors()) %>%
  # Remover variáveis altamente correlacionadas (threshold default 0.9)
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  # Transformar variáveis categóricas em dummy
  step_dummy(all_nominal_predictors()) %>%
  # Padronizar variáveis numéricas
  step_normalize(all_numeric_predictors())

#apenas para mostrar os dados padronizados
df_pre_processado = bake(prep(rec, training = train_data), new_data = train_data)

# ==================================================
# 8. Tunagem de hiperparâmetros
# ==================================================

# fazer a regressão lasso e ridge

# Ridge (alpha = 0)
ridge_model = linear_reg(penalty = tune(), mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Lasso (alpha = 1)
lasso_model = linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Elastic Net (0 < alpha < 1)
elastic_model = linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")


rf_model = rand_forest(       # Define modelo de Random Forest
  trees = 500,                # Número de árvores na floresta
  mtry = tune(),              # Número de variáveis a serem testadas em cada nó (a ser tunado)
  min_n = tune()              # Número mínimo de observações em um nó (a ser tunado)
) %>% 
  set_engine("ranger") %>%    # Usa o engine "ranger" para Random Forest
  set_mode("regression")      # Define o modo como regressão

xgb_model = boost_tree(       # Define modelo XGBoost
  trees = 1000,               # Número máximo de árvores
  tree_depth = tune(),        # Profundidade máxima das árvores (a ser tunado)
  learn_rate = tune(),        # Taxa de aprendizado (a ser tunado)
  loss_reduction = tune()     # Redução mínima da perda para fazer divisão (a ser tunado)
) %>%
  set_engine("xgboost") %>%   # Usa o engine "xgboost"
  set_mode("regression")      # Define o modo como regressão


# ==================================================
# Workflow: combina receita e modelo
# ==================================================

ridge_wf   = workflow() %>% add_recipe(rec) %>% add_model(ridge_model) # Workflow para ridge
lasso_wf   = workflow() %>% add_recipe(rec) %>% add_model(lasso_model)# Workflow para lasso
elastic_wf = workflow() %>% add_recipe(rec) %>% add_model(elastic_model) # Workflow para elastic net
rf_wf  = workflow() %>% add_recipe(rec) %>% add_model(rf_model)    # Workflow para Random Forest
xgb_wf = workflow() %>% add_recipe(rec) %>% add_model(xgb_model)   # Workflow para XGBoost

set.seed(123)  # Garante reprodutibilidade
cv_folds = vfold_cv(train_data, v = 5)  # Cria validação cruzada em 5 folds

# ==================================================
# Criação de grids para tuning de hiperparâmetros
# ==================================================

set.seed(123)  # Garante reprodutibilidade

# Para Ridge e Lasso: apenas penalty (lambda)
# Usamos escala logarítmica (10^-5 até 10^2)
penalty_grid = grid_random(
  penalty(range = c(-5, 2)),  # log10 scale
  size = 10                   # Número de combinações geradas
)

# Para Elastic Net: tunamos penalty e mixture (alpha)
# penalty: 10^-5 a 10^2 (log scale)
# mixture: 0.1 a 0.9 (combinação de L1 e L2)
elastic_grid = grid_random(
  penalty(range = c(-5, 2)),     # escala logarítmica
  mixture(range = c(0.1, 0.9)),  # alpha de 0.1 a 0.9
  size = 10              # Número de combinações geradas
)


# Random Forest:
# - mtry: número de variáveis explicativas consideradas em cada divisão de nó.
# - min_n: número mínimo de observações que um nó deve ter para ser dividido.
# - size: número de combinações de parâmetros geradas aleatoriamente.

rf_grid = grid_random(              # Grid aleatório para Random Forest
  mtry(range = c(1, 8)),            # Intervalo de valores para mtry
  min_n(range = c(2, 10)),          # Intervalo de valores para min_n
  size = 10                          # Número de combinações geradas
)

# XGBoost:
# - tree_depth: profundidade máxima de cada árvore (quanto maior, mais complexa a árvore).
# - learn_rate: taxa de aprendizado (quanto menor, mais lenta e estável a convergência do modelo).
# - loss_reduction: redução mínima da função de perda necessária para realizar uma divisão (regularização para evitar overfitting).
# - size: número de combinações de parâmetros geradas aleatoriamente.


xgb_grid = grid_random(             # Grid aleatório para XGBoost
  tree_depth(range = c(2, 10)),     # Intervalo de profundidade das árvores
  learn_rate(range = c(0.01, 0.3)), # Intervalo da taxa de aprendizado
  loss_reduction(range = c(0, 10)), # Intervalo da redução mínima de perda
  size = 10                          # Número de combinações geradas
)

set.seed(123)  # Garante reprodutibilidade

# ==================================================
# Ajuste dos modelos com validação cruzada
# ==================================================

ridge_tune = tune_grid(ridge_wf, resamples = cv_folds, grid = penalty_grid, metrics = metric_set(rmse, mae)) # Tuning ridge
lasso_tune = tune_grid(lasso_wf,resamples = cv_folds, grid = penalty_grid,metrics = metric_set(rmse, mae)) # tuning lasso
elastic_tune = tune_grid(elastic_wf, resamples = cv_folds, grid = elastic_grid, metrics = metric_set(rmse, mae)) # Tuning elastic net
rf_tune  = tune_grid(rf_wf, resamples = cv_folds, grid = rf_grid, metrics = metric_set(rmse, mae))   # Tuning RF
xgb_tune = tune_grid(xgb_wf, resamples = cv_folds, grid = xgb_grid, metrics = metric_set(rmse, mae)) # Tuning XGBoost

# ==================================================
# Seleção dos melhores hiperparâmetros com base no RMSE
# ==================================================

ridge_tune %>%
  collect_metrics()  %>%
  filter(.metric == "rmse") %>%
  select(penalty, mean, std_err)

lasso_tune %>%
  collect_metrics()  %>%
  filter(.metric == "rmse") %>%
  select(penalty, mean, std_err)

elastic_tune %>%
  collect_metrics()  %>%
  filter(.metric == "rmse") %>%
  select(penalty,mixture , mean, std_err)

rf_tune %>%
  collect_metrics()  %>%
  filter(.metric == "rmse") %>%
  select(mtry, min_n, mean ,std_err)

xgb_tune %>%
  collect_metrics()  %>%
  filter(.metric == "rmse") %>%
  select(tree_depth, learn_rate , loss_reduction , mean ,std_err)


best_ridge   = ridge_tune %>% select_best(metric = "rmse") # Seleciona melhores parâmetros ridge
best_lasso   = lasso_tune %>% select_best(metric = "rmse") # Seleciona melhores parâmetros lasso
best_elastic = elastic_tune %>% select_best(metric = "rmse") # Seleciona melhores parâmetros elastic
best_rf  = rf_tune %>% select_best(metric ="rmse")   # Seleciona melhores parâmetros RF
best_xgb = xgb_tune %>% select_best(metric = "rmse") # Seleciona melhores parâmetros XGB

# ==================================================
# Finaliza workflow com melhores parâmetros
# ==================================================
final_ridge   = finalize_workflow(ridge_wf, best_ridge)  # Workflow ridge final
final_lasso   = finalize_workflow(lasso_wf, best_lasso)  # Workflow lasso final
final_elastic = finalize_workflow(elastic_wf, best_elastic) # Workflow elastic final
final_rf  = finalize_workflow(rf_wf, best_rf)     # Workflow RF final
final_xgb = finalize_workflow(xgb_wf, best_xgb)   # Workflow XGB final

# ==================================================
# Ajusta os modelos finais nos dados de treino
# ==================================================

final_ridge_fit   = fit(final_ridge, data = train_data) # Ajusta ridge
final_lasso_fit   = fit(final_lasso, data = train_data) # Ajusta lasso 
final_elastic_fit = fit(final_elastic, data = train_data) # Ajusta elastic net
final_rf  = fit(final_rf, data = train_data)      # Ajusta Random Forest
final_xgb = fit(final_xgb, data = train_data)     # Ajusta XGBoost

# ==================================================
# 9. Avaliando o Modelo
# ==================================================

# Previsões no conjunto de treino
pred_ridge_treino   = predict(final_ridge_fit, new_data = train_data) %>% bind_cols(train_data)   # Previsões treino Ridge
pred_lasso_treino   = predict(final_lasso_fit, new_data = train_data) %>% bind_cols(train_data)   # Previsões treino Lasso
pred_elastic_treino = predict(final_elastic_fit, new_data = train_data) %>% bind_cols(train_data) # Previsões treino Elastic Net
pred_rf_treino      = predict(final_rf, new_data = train_data) %>% bind_cols(train_data)          # Previsões treino RF
pred_xgb_treino     = predict(final_xgb, new_data = train_data) %>% bind_cols(train_data)         # Previsões treino XGB

# Previsões no conjunto de teste
pred_ridge_teste   = predict(final_ridge_fit, new_data = test_data) %>% bind_cols(test_data)      # Previsões teste Ridge
pred_lasso_teste   = predict(final_lasso_fit, new_data = test_data) %>% bind_cols(test_data)      # Previsões teste Lasso
pred_elastic_teste = predict(final_elastic_fit, new_data = test_data) %>% bind_cols(test_data)    # Previsões teste Elastic Net
pred_rf_teste      = predict(final_rf, new_data = test_data) %>% bind_cols(test_data)             # Previsões teste RF
pred_xgb_teste     = predict(final_xgb, new_data = test_data) %>% bind_cols(test_data)            # Previsões teste XGB


# ==================================================
# Calcula métricas para dados de treino
# ==================================================

metricas_ridge_treino   =  pred_ridge_treino %>%  metrics(truth = charges, estimate = .pred)  %>% mutate(Modelo = "Ridge")
metricas_lasso_treino   =  pred_lasso_treino %>%  metrics(truth = charges, estimate = .pred)  %>% mutate(Modelo = "Lasso")
metricas_elastic_treino =  pred_elastic_treino %>%  metrics(truth = charges, estimate = .pred) %>% mutate(Modelo = "Elastic Net")
metricas_rf_treino  = pred_rf_treino  %>%  metrics(truth = charges, estimate = .pred) %>% mutate(Modelo = "Random Forest")
metricas_xgb_treino = pred_xgb_treino %>%  metrics(truth = charges, estimate = .pred) %>% mutate(Modelo = "XGBoost")

resultados_treino = bind_rows(
  metricas_ridge_treino,
  metricas_lasso_treino,
  metricas_elastic_treino,
  metricas_rf_treino,
  metricas_xgb_treino
) %>%
  filter(.metric == "rmse") %>%
  arrange(.estimate) # Junta e ordena pelo RMSE

# ==================================================
# Calcula métricas para dados de teste
# ==================================================

metricas_ridge_teste   = pred_ridge_teste %>%  metrics(truth = charges, estimate = .pred)   %>% mutate(Modelo = "Ridge")
metricas_lasso_teste   = pred_lasso_teste %>%  metrics(truth = charges, estimate = .pred)   %>% mutate(Modelo = "Lasso")
metricas_elastic_teste = pred_elastic_teste %>%  metrics(truth = charges, estimate = .pred) %>% mutate(Modelo = "Elastic Net")
metricas_rf_teste  = pred_rf_teste %>%  metrics(truth = charges, estimate = .pred) %>% mutate(Modelo = "Random Forest")
metricas_xgb_teste = pred_xgb_teste %>%  metrics(truth = charges, estimate = .pred) %>% mutate(Modelo = "XGBoost")

# Junta e ordena pelo RMSE (quanto menor, melhor)
resultados_teste = bind_rows(
  metricas_ridge_teste,
  metricas_lasso_teste,
  metricas_elastic_teste,
  metricas_rf_teste,
  metricas_xgb_teste
) %>%
  filter(.metric == "rmse") %>%
  arrange(.estimate)


# Adiciona coluna indicando etapa
resultados_teste$etapa = "Teste"
resultados_treino$etapa = "Treino"

# Junta resultados treino e teste
View(resultados_treino %>% bind_rows(resultados_teste) %>% arrange(desc(Modelo), desc(etapa) ))


# modelo final, sendo recalculado pra base toda, com os melhores hiperparâmetros
modelo_final = final_rf %>% fit(data = df)

#salvando o modelo

saveRDS(modelo_final, "modelo_final.rds")

# ==================================================
# Simulação de Produção
# ==================================================


# carregando o modelo
workflow_prod = readRDS("modelo_final.rds")

dados_prod = read.csv("dados_producao.csv")

# função para automatizar a predição
realizar_predicao = function(workflow, dados){

  previsoes = predict(workflow, new_data = dados) %>%
  bind_cols(dados)
  
  return(previsoes)
}

# Previsões diretas
previsoes = realizar_predicao(workflow_prod,dados_prod)
