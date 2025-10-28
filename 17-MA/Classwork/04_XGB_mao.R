# pacotes
library(tidymodels)
library(tidyverse)
library(xgboost)
library(DiagrammeR)
# carregando dados do exemplo feito a mao
y <- c(-10,-7,7,8)
x <- c(10,35,20,25)
dados <- as.data.frame(cbind(y,x))

# especificando o modelo
modelo_xgb <- boost_tree(
  mode = "regression",  # que tipo de arvore eh: regression ou classification
  mtry = 1, # numero de variaveis em cada arvore
  sample_size = 1, # percentual das linhas sorteadas por arvore
  min_n = 1, #numero minimo que um no precisa ter para quebrar
  tree_depth = 2, # profundidade da arvore
  loss_reduction = 90, # gamma, conhecido como loss reduction
  trees = 1, #numero de arvores
  learn_rate = 0.3 # taxa de aprendizado
) %>%
  set_engine("xgboost", lambda = 0) # aqui você adiona o valor da regularizacao lambda

# apos especificar, ajustar
ajustando_xgb <- fit(modelo_xgb, y ~ x, data = dados)
ajustando_xgb

# criando uma coluna predita e residuo
dados_pred_res <- dados %>% 
  mutate(
  predicao = predict(ajustando_xgb, dados),
  erro = y-predicao
)

#plotando as arvores criadas
xgb.plot.tree(model=ajustando_xgb$fit)

