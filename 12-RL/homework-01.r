library(tidyverse)


estudante <- c(2,6,8,8,12,16,20,20,22,26)
pizza <- c(55,105,88,118,117,137,157,169,149,202)
dados <- cbind(estudante,pizza)
dados_pizzaria <- as.data.frame(dados)

view(dados_pizzaria)

plot(dados_pizzaria$estudante, dados_pizzaria$pizza)

dados_pizzaria['y_barra'] <- mean(dados_pizzaria$pizza)

plot(dados_pizzaria$pizza, dados_pizzaria$y_barra)

reg_intercepto <- lm(pizza ~ 1, data=dados_pizzaria)
summary(reg_intercepto)


reg_simples <- lm(pizza ~ estudante, data=dados_pizzaria)
summary(reg_simples)

