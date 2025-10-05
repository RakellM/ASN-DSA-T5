#Vamos começar a brincadeira#####################

#1. Carregando Pacotes#####
library(readxl)
library(haven)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(rcompanion) # cramersV
library(fastDummies)
library(jtools)


# Brincando com o exemplo de compra e nao compra#########

#2. importando a base de dados
idade_renda_compra <- read_excel("dados/idade_renda_compra.xlsx")
View(idade_renda_compra)

#3. criando as variaveis dummies
idade_renda_compra_dm <- dummy_columns(.data = idade_renda_compra)

#4. como criamos todas as dummies,vamos deixar apenas as que levaremos pro modelo
idade_renda_compra_dm <- idade_renda_compra_dm %>% 
  select(-compra, - renda, -compra_nao_compra, -renda_baixa)

#4. fazendo o modelo apenas com renda ALTA
modelo_log <- glm(compra_compra ~ renda_alta ,family=binomial(link = "logit"), data=idade_renda_compra_dm)
summary(modelo_log)

#5. interpretando o parametro
exp(modelo_log$coefficients)

#6. fazendo o modelo com renda + idade (lembre que aqui é apenas para discussao)
# como nao existe uma relacao linear, deveriamos agrupar a variavel idade

modelo_log2 <- glm(compra_compra ~ renda_alta+idade,family=binomial(link = "logit"), data=idade_renda_compra_dm)
summary(modelo_log2)

#7. interpretando os parametros
exp(modelo_log2$coefficients)

#8. interpretando os parametros com mudanca de espacamento na variavel numerica
exp(10*modelo_log2$coefficients[3])

#9. simulando uma nova base, apenas para discussao dos p-valores
# estamos criando a renda media que tem o mesmo comportamento da alta
gambs1 <- idade_renda_compra %>% 
  filter(renda == "alta") %>% 
  mutate(renda= "media")
gambs_discussao <- rbind(idade_renda_compra, gambs1)

#10. criando as variaveis dummies da gambs para discussao
gambs_discussao_dm <- dummy_columns(.data = gambs_discussao)


teste <- gambs_discussao_dm %>%
  group_by(renda) %>%
  summarise(media_y = mean(compra_compra))
teste

#11. como criamos todas as dummies,vamos deixar apenas as que levaremos pro modelo
# aqui estou deixando todas as dummies da variavel renda - a qual vamos explorar
# pois assim podemos ir variando a casela de referencia nos modelos que brincaremos
gambs_discussao_dm <- gambs_discussao_dm %>% 
  select(-compra, - renda, - idade, -compra_nao_compra)

#12. vamos construir o modelo para ver o comportamento do p-valor
# vamos fazer retirando cada hora uma dummie (ou seja, cada hora uma sera  casela de refeencia! 
# vou comecar tirando baixa!
# voce ja sabe o que vai acontecer como p-valor das dummies que forem para o modelo?
modelo_log3 <- glm(compra_compra ~ renda_alta + renda_media,family=binomial(link = "logit"), data=gambs_discussao_dm)
summary(modelo_log3)

#13. curiosidade observando intervalo de confianca para os parametros
confint(modelo_log3, level = 0.95)
plot_summs(modelo_log3, colors = "#4400FF")

#14. e se agora tirassemos renda_alta, o que iria acontecer?
modelo_log4 <- glm(compra_compra ~ renda_baixa + renda_media,family=binomial(link = "logit"), data=gambs_discussao_dm)
summary(modelo_log4)

#15. observe os intervalos
confint(modelo_log4, level = 0.95)
plot_summs(modelo_log4, colors = "#4400FF")

#16. e se eu tirar a renda_media?
modelo_log5 <- glm(compra_compra ~ renda_baixa + renda_alta,family=binomial(link = "logit"), data=gambs_discussao_dm)
summary(modelo_log5)

#17. observe os intervalos
confint(modelo_log5, level = 0.95)
plot_summs(modelo_log5, colors = "#4400FF")
