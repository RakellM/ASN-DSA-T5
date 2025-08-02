#Exercicio PRECOS - sera que agora vai?####################

#' voce se lembra do problema do preco da casa?
#' Nos tentamos fazer o modelo com as variaveis direto e analise de residuo quebra
#' Tentamos com o log das variaveis e analise de residuo tambem quebra
#' Existe uma outra forma... transformar o y
#' Normalmente isso ajuda a normalizar os residuos
#' Isso foi eh tema de estudo do dono do artigo que revolucionou sobre o tema
#' Link: https://www.ime.usp.br/~abe/lista/pdfQWaCMboK68.pdf

#1. Carregando Pacotes#####
library(readr)
library(openxlsx) #biblioteca para escrever arquivo em excel
library(haven)
library(readxl)
library(tidyverse)
library(yardstick) #biblioteca para calcular medidas de erro
library(lmtest) # calcula o teste de homogeneidade de variancia
library(car) # calcula vif
library(ggraph)
library(plotly)
library(ggstance)
library(jtools)
library(olsrr)
library(PerformanceAnalytics)
library(correlation)
library(fastDummies)
library(lazyeval) # para a analise de interacao


#2. Carregando os dados#####
# fazendo a leitura do arquivo ja em R que estao na pasta dados

precos <- readRDS("dados/precos.rds")

# nesse momento pegando apenas as variaveis numericas para brincar

precos <- precos %>% 
  select(-Heating_QC, -Season_Sold)

#Como ja tentamos muitas coisas, vamos tentar transformar o y#######
#3. Transformacao Box-Cox#######

#entenda como o transformador funciona 
#3.T1 - Descobrindo o Lambda########

lambda <- powerTransform(precos$SalePrice) #função powerTransform do pacote car#
lambda

#3.T2 - Criando y transformado########

precos$BC_SalePrice <- (((precos$SalePrice ^ lambda$lambda) - 1) / 
                          lambda$lambda)

#3.T3 - Criando modelo step por p-valor########

modelo_BC_SalePrice_full <- lm(BC_SalePrice ~ . -SalePrice, data = precos)
summary(modelo_BC_SalePrice_full)

modelo_BC_SalePrice_setp_pvalor <- step(modelo_BC_SalePrice_full, k = 3.841459)
summary(modelo_BC_SalePrice_setp_pvalor)

#3.T4 - Fazendo a Analise de Residuo########
# plotando o ajustado X residuo 

plot(fitted(modelo_BC_SalePrice_setp_pvalor),residuals(modelo_BC_SalePrice_setp_pvalor),xlab="Valores Ajustados",ylab="Resíduos")
abline(h=0)

# teste de normalidade do residuo 
# h0: os dados sao normais
# h1: os dados nao sao normais

shapiro.test(modelo_BC_SalePrice_setp_pvalor$residuals)

#Seguimos sem modelo!!! huauhahua##################
# Mas nao quer dizer que porque nao resolveu o problema em precos
# que quer dizer que isso nao funciona
# muito pelo contrario

#Transformação Box-Cox - Modelos Nao Lineares#######
# isso eh muito util para Modelos onde nao existe relacao linear

#Vamos praticar em outro exemplo#######
#1. Carregando os dados#####
# fazendo a leitura do arquivo ja em R que estao na pasta dados

Notas <- read_excel("dados/dados_NOTAS.xlsx")
View(Notas)

#2. Visualizando#####

xy <- ggplot()
xy <- xy + geom_point(data=Notas, aes(x=horas_estudo, y=desempenho), color = "black" )
xy

#3. Ignora e faz um modelo linear#####

modelo_linear <- lm(desempenho ~ horas_estudo, data = Notas)
summary(modelo_linear)

Notas <- Notas %>% 
  mutate(ychapeu_linear = predict(modelo_linear, data.frame(horas_estudo = horas_estudo)))

#4. Verificando desempenho#####
bora <- ggplot()
bora <- bora + geom_point(data=Notas, aes(x=horas_estudo, y=desempenho))
bora
bora <- bora + geom_point(data=Notas, aes(x=horas_estudo, y=ychapeu_linear, color = "ychapeu_linear") )
bora

#5. Aplicando Transformacao Box-Cox#######

#5.T1 - Descobrindo o Lambda########

lambda <- powerTransform(Notas$desempenho) 
lambda

#5.T2 - Criando y transformado########

Notas$BC_Desempenho <- (((Notas$desempenho ^ lambda$lambda) - 1) / 
                          lambda$lambda)

#5.T3 - Criando modelo step por p-valor########

modelo_nao_linear <- lm(BC_Desempenho ~ horas_estudo, data = Notas)
summary(modelo_nao_linear)

#5.T4 - Fazendo a Analise de Residuo########
# plotando o ajustado X residuo 

plot(fitted(modelo_nao_linear),residuals(modelo_nao_linear),xlab="Valores Ajustados",ylab="Resíduos")
abline(h=0)

# teste de normalidade do residuo 
# h0: os dados sao normais
# h1: os dados nao sao normais

shapiro.test(modelo_nao_linear$residuals)


#6. Visualizando a Arte#####
# primeiro criando a ychapeu desse modelo 
# nesse caso nao eh tao simples nem direto
# lembra que estamos modelando o Y transformado?
# Entao nosso predict estima o transformado de Y
# Para trazermos para o Y verdadeiro, temos que aplicar
# a transformacao!!!
# formula: (((BC_y * lambda) + 1)) ^ (1 / lambda)
Notas <- Notas %>% 
  mutate(ychapeu_transformado_nao_linear = predict(modelo_nao_linear, data.frame(horas_estudo = horas_estudo)),
         ychapeu_nao_linear = (((ychapeu_transformado_nao_linear * lambda$lambda) + 1)) ^ (1 / lambda$lambda) )

bora <- ggplot()
bora <- bora + geom_point(data=Notas, aes(x=horas_estudo, y=desempenho))
bora
bora <- bora + geom_point(data=Notas, aes(x=horas_estudo, y=ychapeu_linear, color = "ychapeu_linear") )
bora
bora <- bora + geom_point(data=Notas, aes(x=horas_estudo, y=ychapeu_nao_linear, color = "ychapeu_nao_linear") )
bora

#7. Visualizando o desempenho#####
eficiencia <- ggplot()
eficiencia <- eficiencia + geom_point(data=Notas, aes(x=desempenho, y=desempenho), color="black")
eficiencia
eficiencia <- eficiencia + geom_point(data=Notas, aes(x=ychapeu_linear, y=desempenho), color="red")
eficiencia
eficiencia <- eficiencia + geom_point(data=Notas, aes(x=ychapeu_nao_linear, y=desempenho), color="green")
eficiencia


#melhorou mas ainda nao passou na normalidade
# esse modelo nao eh a melhor opcao

#Vamos praticar em outro exemplo#######
#1. Carregando os dados#####
# fazendo a leitura do arquivo ja em R que estao na pasta dados
load(file = "dados/bebes.RData")

#Estatisticas univariadas

summary(bebes)

#2. Visualizando#####
xy <- ggplot()
xy <- xy + geom_point(data=bebes, aes(x=idade, y=comprimento), color = "black" )
xy

#3. Ignora e faz um modelo linear#####
modelo_linear <- lm(comprimento ~ idade, data = bebes)
summary(modelo_linear)

bebes <- bebes %>% 
  mutate(ychapeu_linear = predict(modelo_linear, data.frame(idade = idade)))

#4. Verificando desempenho#####
bora <- ggplot()
bora <- bora + geom_point(data=bebes, aes(x=idade, y=comprimento))
bora
bora <- bora + geom_point(data=bebes, aes(x=idade, y=ychapeu_linear, color = "ychapeu_linear") )
bora

#5. Aplicando Transformacao Box-Cox#######

#5.T1 - Descobrindo o Lambda########
lambda <- powerTransform(bebes$comprimento) 
lambda

#5.T2 - Criando y transformado########
bebes$BC_Comprimento <- (((bebes$comprimento ^ lambda$lambda) - 1) / 
                           lambda$lambda)

#5.T3 - Criando modelo step por p-valor########
modelo_nao_linear <- lm(BC_Comprimento ~ idade, data = bebes)
summary(modelo_nao_linear)

#5.T4 - Fazendo a Analise de Residuo########
# plotando o ajustado X residuo 

plot(fitted(modelo_nao_linear),residuals(modelo_nao_linear),xlab="Valores Ajustados",ylab="Resíduos")
abline(h=0)

# teste de normalidade do residuo 
# h0: os dados sao normais
# h1: os dados nao sao normais
shapiro.test(modelo_nao_linear$residuals)


# fazendo o histograma de uma forma mais bonita
bebes %>%
  mutate(residuos = modelo_nao_linear$residuals) %>%
  ggplot(aes(x = residuos)) +
  geom_histogram(color = "white", 
                 fill = "#4400FF", 
                 bins = 30,
                 alpha = 0.6) +
  labs(x = "Resíduos",
       y = "Frequência") + 
  theme_bw()

# adicionando uma curva da normal por cima
bebes %>%
  mutate(residuos = modelo_nao_linear$residuals) %>%
  ggplot(aes(x = residuos)) +
  geom_histogram(aes(y = ..density..), 
                 color = "white", 
                 fill = "#4400FF", 
                 bins = 30,
                 alpha = 0.6) +
  stat_function(fun = dnorm, 
                args = list(mean = mean(modelo_nao_linear$residuals),
                            sd = sd(modelo_nao_linear$residuals)),
                size = 2, color = "grey30") +
  scale_color_manual(values = "grey50") +
  labs(x = "Resíduos",
       y = "Frequência") +
  theme_bw()

#Teste de homogeneidade de variancia Breusch-Pagan test
#h0: as variancias dos erros sao iguais (homoscedasticidade)
#h1: as variancias dos erros nao sao iguais (heteroscedasticidade)
bptest(modelo_nao_linear) #library lmtest

#nao passou em nada tb! =/

#6. Visualizando a Arte#####
# primeiro criando a ychapeu desse modelo 
# nesse caso nao eh tao simples nem direto
# lembra que estamos modelando o Y transformado?
# Entao nosso predict estima o transformado de Y
# Para trazermos para o Y verdadeiro, temos que aplicar
# a transformacao!!!
# formula: (((BC_y * lambda) + 1)) ^ (1 / lambda)
bebes <- bebes %>% 
  mutate(ychapeu_transformado_nao_linear = predict(modelo_nao_linear, data.frame(idade = idade)),
         ychapeu_nao_linear = (((ychapeu_transformado_nao_linear * lambda$lambda) + 1)) ^ (1 / lambda$lambda) )

bora <- ggplot()
bora <- bora + geom_point(data=bebes, aes(x=idade, y=comprimento))
bora
bora <- bora + geom_point(data=bebes, aes(x=idade, y=ychapeu_linear, color = "ychapeu_linear") )
bora
bora <- bora + geom_point(data=bebes, aes(x=idade, y=ychapeu_nao_linear, color = "ychapeu_nao_linear") )
bora

#7. Visualizando o desempenho#####
eficiencia <- ggplot()
eficiencia <- eficiencia + geom_point(data=bebes, aes(x=comprimento, y=comprimento), color="black")
eficiencia
eficiencia <- eficiencia + geom_point(data=bebes, aes(x=ychapeu_linear, y=comprimento), color="red")
eficiencia
eficiencia <- eficiencia + geom_point(data=bebes, aes(x=ychapeu_nao_linear, y=comprimento), color="green")
eficiencia

