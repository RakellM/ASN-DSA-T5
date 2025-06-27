#Vamos começar a brincadeira#####################

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



#2. Brincando com o exemplo das pizzas#########

# criando a base de dados
estudante <- c(2,6,8,8,12,16,20,20,22,26)
pizza <- c(55,105,88,118,117,137,157,169,149,202)
dados <- cbind(estudante,pizza)
dados_pizzaria <- as.data.frame(dados)

#2a. Conhecendo o meu quadro a desenhar#########
# variavel dependente (pizza)
summary(dados_pizzaria$pizza)
sd(dados_pizzaria$pizza)
hist(dados_pizzaria$pizza)
boxplot(dados_pizzaria$pizza)

#2b. Testando a variavel estudante (x) para explicar pizza (y)#########
# fazendo um plot simples da variavel x e y
# para ver se da vontade de passar a reta
plot(dados_pizzaria$estudante, dados_pizzaria$pizza)

#2c. Tracando um modelo de regressao######
# deu vontade, vou tentar fazer o modelo de regressao
# executando o modelo de regressao linear simples
reg_simples <- lm(pizza ~ estudante, data=dados_pizzaria)
summary(reg_simples)

#2d. Verificando o quanto o modelo ficou bom######
# quero entender o quanto meu desenho ficou bom
# primeiro vou criar esse chute
meus_desenhos <- dados_pizzaria %>% 
  mutate(chute_media = mean(dados_pizzaria$pizza),
         chute_modelo = reg_simples$fitted.values)

#2d.. Visualmente######
# agora vou ver o como ficou o desenho
# agora vou plotar os chutes junto com o desenho real
arte <- ggplot()
arte <- arte + geom_point(data=meus_desenhos, aes(x=pizza, y=pizza), color = "black" ) + 
  labs(title = "",
       x = "nosso_desenho",
       y = "real_arte",
       color = "green")
arte
arte <- arte + geom_point(data=meus_desenhos, aes(x=chute_media, y=pizza), color = "gray" )
arte
arte <- arte + geom_point(data=meus_desenhos, aes(x=chute_modelo, y=pizza), color = "blue" )
arte

# OLHA QUE DO PIRU ##############################
#Grafico didatico para visualizar o conceito de R²
ggplotly(
  ggplot(meus_desenhos, aes(x = estudante, y = pizza)) +
    geom_point(color = "#39568CFF", size = 2.5) +
    geom_smooth(aes(color = "Fitted Values"),
                method = "lm", se = F, size = 2) +
    geom_hline(yintercept = 129.7, color = "grey50", size = .5) +
    geom_segment(aes(color = "Ychapéu - Ymédio", x = estudante, xend = estudante,
                     y = chute_modelo, yend = mean(pizza)), size = 0.7, linetype = 2) +
    geom_segment(aes(color = "Erro = Y - Ychapéu", x = estudante, xend = estudante,
                     y = pizza, yend = chute_modelo), size = 0.7, linetype = 3) +
    labs(x = "Estudante",
         y = "Pizza") +
    scale_color_manual("Legenda:",
                       values = c("#5457FF", "black", "pink")) +
    theme_classic()
)

#2d.. Atraves das metricas de qualidade de ajuste######

# Calculando MSE e RMSE do modelo

# Media de erro do meu modelo ao quadrado (MSE - Mean Square Error)
mse_modelo <- mean((meus_desenhos$pizza - meus_desenhos$chute_modelo)^2)
mse_modelo
# Raiz da media de erro do meu modelo ao quadrado (RMSE - Root Mean Square Error)
rmse_modelo <- sqrt(mse_modelo)
rmse_modelo

# Media de erro do modelo media (a famosa variancia)
mse_usando_media <- mean((meus_desenhos$pizza - meus_desenhos$chute_media)^2)
mse_usando_media
# Raiz da media de erro do modelo media (o famoso desvio padao)
rmse_usando_media <- sqrt(mse_usando_media)
rmse_usando_media


#3. Brincando e aprendendo com o problema Bodyfat#####################################################
# exercicio BODYFAT
# importando CSV que foi baixado em: http://staff.pubhealth.ku.dk/~tag/Teaching/share/data/Bodyfat.html
Bodyfat <- read_csv("dados/Bodyfat.csv")
# tirando uma variavel que nao sera usada (contexto negocio)
Bodyfat <- Bodyfat %>% 
  select(-Density)
View(Bodyfat)

#3a. Conhecendo minha variavel y#####################################################

# Faca uma breve descritiva da variavel y

# variavel dependente (bodyfat)
summary(Bodyfat$bodyfat)
sd(Bodyfat$bodyfat)
hist(Bodyfat$bodyfat)
boxplot(Bodyfat$bodyfat)
#' comentarios:
#' eh a minha variavel mais importante, pois eh ela que
#' eu vou prever, sendo assim ela deve ter qualidade
#' (bom, todas outras tambem, uhahuahua)
#' aqui, poderia me surgir um questionamento sobre o outlier
#' e quem sabe eu descubra que isso foi um erro de digitacao
#' e por ser a variavel resposta, melhor deletar esse individuo
#' do que tentar chutar um valor para ele
#' esse eh o tipo de pergunta que eu faria para o negocio se faz sentido

#4. Assumindo que exista apenas a variavel X = Wrist###############################
# as demais ficarao para depois
# um passo de cada vez!

#FACA VOCE - Explore a variavel Wrist###########################
# a variavel explicativa (Wrist)










#FACA VOCE - Visualize a relação entre X e Y###########################
# dispersao entre elas





#FACA VOCE - Calcule a correlacao entre X e Y###########################
# correlacao entre as duas variaveis


#FACA VOCE - Gere uma equacao de regressao linear simples entre bodyfat e Wrist###########################
#modelo MRLS - nome do objeto reg_gordura_wrist




#4a. Verificando o quanto o modelo ficou bom######
#FACA VOCE - Crie a tentativa media e a modelo###########################
# crie uma nova base chamada tentativas com a base original 
# mais o chute_media e o chute_modelo (exatamente com esses nomes)
tentativas <- Bodyfat %>% 
  mutate(chute_media = mean(Bodyfat$bodyfat),
         chute_modelo = reg_gordura_wrist$fitted.values)




#4a.. Visualmente######
#FACA VOCE - Plote a ARTE e os seus desenhos###########################

# agora vou plotar os chutes junto com o desenho real













#4a... Atraves das metricas de qualidade de ajuste######
#FACA VOCE - Calcule MSE e RMSE desses dois chutes###########################

# Calculando MSE e RMSE do modelo

# Media de erro do meu modelo ao quadrado (MSE - Mean Square Error)
mse_modelo <- mean((tentativas$bodyfat - tentativas$chute_modelo)^2)
mse_modelo
# Raiz da media de erro do meu modelo ao quadrado (RMSE - Root Mean Square Error)
rmse_modelo <- sqrt(mse_modelo)
rmse_modelo

# Media de erro do modelo media (a famosa variancia)
mse_usando_media <- mean((tentativas$bodyfat - tentativas$chute_media)^2)
mse_usando_media
# Raiz da media de erro do modelo media (o famoso desvio padao)
rmse_usando_media <- sqrt(mse_usando_media)
rmse_usando_media



#5. Assumindo que existam todas as variáveis X´s###############################
# Regressao Linear Multipla 
# bodyfat   Age Weight Height  Neck Chest Abdomen   Hip Thigh  Knee Ankle Biceps
# fazendo o modelo com todas as variaveis
# nome do objeto reg_gordura_full
reg_gordura_full <- lm(bodyfat ~ ., data=Bodyfat)
summary(reg_gordura_full)

#REFLITA###########################
#' o que aconteceu com o modelo?
#' veja os testes estatisticos


#6. Criando o modelo com seleção de variavel###############################
# regressao por forward com a regra de AIC
# para o forward precisa primeiro ter o modelo sem nenhuma variavel
# ou seja, apenas o modelo media (b0)
reg_gordura_nula <- lm(bodyfat ~ 1, data=Bodyfat)
summary(reg_gordura_nula)

# com ele podemos pedir o step saindo dele, ate aquele modelo com todas as vars
forw <- step(reg_gordura_nula, scope=list(lower=reg_gordura_nula, upper=reg_gordura_full), direction = "forward")
summary(forw)

# regressao por backward com a regra de AIC
backw <-  step(reg_gordura_full, direction = "backward")
summary(backw)

# regressao por stepwise com a regra de AIC
stepw <- step(reg_gordura_full, direction = "both")
summary(stepw)

# regressao por stepwise com a regra de p-valor
stepw_p <- step(reg_gordura_full, k = 3.841459)
summary(stepw_p)

#6a. Verificando o quanto os modelos ficaram bons######
#FACA VOCE - Crie as novas tentativas###########################
# adicione nas nossa base tentativas, os novos chutes 
# respeite os nomes dos chutes
# chute_modelo_full, chute_modelo_step_aic, chute_modelo_step_pvalor





#FAÇA VOCE - Plote a ARTE e os seus desenhos###########################

# agora vou plotar os chutes junto com o desenho real


















#Calcule MSE e RMSE desses 3 novos chutes###########################

# pegando a medida RMSE de cada modelo
# usando a biblioteca yardstick
# Metricas de erro
rmse(tentativas, truth = bodyfat, estimate = chute_media)
rmse(tentativas, truth = bodyfat, estimate = chute_modelo)
rmse(tentativas, truth = bodyfat, estimate = chute_modelo_full)
rmse(tentativas, truth = bodyfat, estimate = chute_modelo_step_aic)
rmse(tentativas, truth = bodyfat, estimate = chute_modelo_step_pvalor)

mae(tentativas, truth = bodyfat, estimate = chute_media)
mae(tentativas, truth = bodyfat, estimate = chute_modelo)
mae(tentativas, truth = bodyfat, estimate = chute_modelo_full)
mae(tentativas, truth = bodyfat, estimate = chute_modelo_step_aic)
mae(tentativas, truth = bodyfat, estimate = chute_modelo_step_pvalor)


#7. Pausa para discussao sobre os parametros###############################

#7a. Justificando porque nao eh bom tirar b0 do modelo###############################
# Para entender isso melhor, vamos pegar nossa regressao simples
# onde tinha-mos apenas a variavel Wrist
summary(reg_gordura_wrist)

# veja como fica nosso modelo no desenho Y contra X
gr <- ggplot()
gr <- gr + geom_point(data=tentativas, aes(x=Wrist, y=bodyfat) )
gr <- gr + geom_line(data=tentativas, aes(x=Wrist, y=chute_modelo))
gr

# agora vamos zerar o intercepto e ver a equacao estimada
reg_gordura_wrist_sem_b0 <- lm(Bodyfat$bodyfat ~ Bodyfat$Wrist - 1)
summary(reg_gordura_wrist_sem_b0)

tentativas <- tentativas %>% 
  mutate(chute_modelo_sem_b0 = reg_gordura_wrist_sem_b0$fitted.values )

# adicionando a reta desse novo "modelo"

gr <- ggplot()
gr <- gr + geom_point(data=tentativas, aes(x=Wrist, y=bodyfat) )
gr <- gr + geom_line(data=tentativas, aes(x=Wrist, y=chute_modelo, color="Modelo com Intercepto"))
gr
gr <- gr + geom_line(data=tentativas, aes(x=Wrist, y=chute_modelo_sem_b0, color="Modelo sem Intercepto"))
gr

# outra forma de ver o mesmo desenho
ggplot(tentativas, aes(x = Wrist, y = bodyfat)) +
  geom_point(color = "#39568CFF", size = 2.5) +
  geom_smooth(aes(color = "Modelo com Intercepto"),
              method = "lm", se = F, size = 1.5) +
  geom_segment(aes(color = "Modelo sem Intercepto",
                   x = min(Wrist),
                   xend = max(Wrist),
                   y = reg_gordura_wrist_sem_b0$coefficients[1]*min(Wrist),
                   yend = reg_gordura_wrist_sem_b0$coefficients[1]*max(Wrist)),
               size = 1.5) +
  labs(x = "Wrist",
       y = "Bodyfat") +
  scale_color_manual("Legenda:",
                     values = c("grey50", "#1F968BFF")) 

#mensure o erro dos dois, rapidamente
#modelo com intercepto
rmse(tentativas, truth = bodyfat, estimate = chute_modelo)
#modelo sem intercepto
rmse(tentativas, truth = bodyfat, estimate = chute_modelo_sem_b0)

#7b. Comparar efeitos e variacoes###############################
# outra forma de ver a saida da nossa regressao
export_summs(stepw_p, scale = F, digits = 5)

# desenhando o intervalo dos parametros
confint(stepw_p, level = 0.95) # significancia 0,05
plot_summs(stepw_p, colors = "#440154FF") #funcao plot_summs do pacote ggstance

# desenhando o intervalo dos parametros (para comparacao)
plot_summs(stepw_p, scale = TRUE, colors = "#440154FF")

# fazendo uma firulinha de desenhar a distribuicao normal
plot_summs(stepw_p, scale = TRUE, plot.distributions = TRUE,
           inner_ci_level = .95, colors = "#440154FF")

# observado a diferenca do modelo completo para o modelo stepwise por p-valor
plot_summs(reg_gordura_full, stepw_p, scale = TRUE, plot.distributions = TRUE,
           inner_ci_level = .95, colors = c("#FDE725FF", "#440154FF"))

#8. Analise de Residuo###############################
#8a. Plotando o grafico de Residuo x Ajustado########
# ajustado X resíduo
plot(fitted(stepw_p),residuals(stepw_p),xlab="ValoresAjustados",ylab="Resíduos")
abline(h=0)
# qqplot
qqnorm(residuals(stepw_p), ylab="Resíduos")
qqline(residuals(stepw_p))

#8b. Fazendo Teste de normalidade de Shapiro############
# teste de normalidade
# h0: os dados sao normais
# h1: os dados nao sao normais
shapiro.test(stepw_p$residuals)

# visualizando o histograma
hist(stepw_p$residuals)

# fazendo o histograma de uma forma mais bonita
Bodyfat %>%
  mutate(residuos = stepw_p$residuals) %>%
  ggplot(aes(x = residuos)) +
  geom_histogram(color = "white", 
                 fill = "#440154FF", 
                 bins = 10,
                 alpha = 0.6) +
  labs(x = "Resíduos",
       y = "Frequência") + 
  theme_bw()

# adicionando uma curva da normal por cima
Bodyfat %>%
  mutate(residuos = stepw_p$residuals) %>%
  ggplot(aes(x = residuos)) +
  geom_histogram(aes(y = ..density..), 
                 color = "white", 
                 fill = "#440154FF", 
                 bins = 10,
                 alpha = 0.6) +
  stat_function(fun = dnorm, 
                args = list(mean = mean(stepw_p$residuals),
                            sd = sd(stepw_p$residuals)),
                size = 2, color = "grey30") +
  scale_color_manual(values = "grey50") +
  labs(x = "Resíduos",
       y = "Frequência") +
  theme_bw()

#8c. Fazendo o Teste de homogeneidade de variancia############
#teste de homogeneidade de variancia Breusch-Pagan test
#h0: as variancias dos erros sao iguais (homoscedasticidade)
#h1: as variancias dos erros nao sao iguais (heteroscedasticidade)
bptest(stepw_p) #library lmtest

#9. Pausa para discussao sobre MULTICOLINEARIDADE###############################

#9a. Primeiro vamos entender o maleficio###############################

# para falarmos sobre isso trouxe uma outra base de dados
# onde o objetivo eh prever o tempo de viagem
# dado o numero de milhas percorridas e galoes gastos
galoes <- read_excel("dados/Galoes.xlsx")
galoes

# vamos direto ao ponto
# fazer o modelo de regressao
reg_galoes <- lm(Tempo_viagem ~ ., data=galoes)
summary(reg_galoes)
# observe o teste F (pelo menos uma variavel presta)
# observe o teste t (nenhuma variavel presta)
# efeitos de multicolinearidade sao um problema
# causam confusao e podem inverter valores de parametros

#9b. Entendendo como evitar###############################

# Bom, sem duvida significa nao colocar variaveis (X´s)
# altamente correlacionadas entre si e multivariadamente tambem

# pensando em verificar isso, pode-se estudar a relacao
# entre as variaveis X´s

chart.Correlation((galoes), histogram = TRUE)

# esta visivel que o X=milhas eh a mesma coisa que o X=galoes
# e ai que mora o problema
# sendo assim, deve-se validar isso antes de colocar no modelo
# e escolher aquelas variaveis que facam mais sentido
# nessa historia aqui, entao eu tiraria a variavel galoes
# pois milhas tem uma relacao ainda mais forte no y

#9c. Entendendo a situacao do modelo Bodyfat stepwise criterio AIC###############################

# repare que existem muitas variaveis X´s com forte 
# relacoes com outras variaveis X´s
chart.Correlation((Bodyfat[2:14]), histogram = TRUE)

# no modelo que fizemos pelo stepwise criterio AIC
summary(stepw)
# tinhamos essas variaveis no modelo
# observe que entre elas, existem altas correlacoes
vars_modelo_stepw <- Bodyfat %>% 
  select(Age,Weight,Neck,Abdomen, Hip, 
         Thigh,Forearm, Wrist)
chart.Correlation(vars_modelo_stepw, histogram = TRUE)

# olha o medo que da uma vez que entendemos que pode 
# causar desconfianca sobre os parametros

# essa eh uma forma
# mas voce concorda que poderia existir uma relacao
# multipla entre as variaveis X´s?
# e foi ai que criaram o VIF
# o quanto eh o R2 da variavel X1 assumundo o papel de Y
# com todas as outras X´s?
# se esse R2 for alto, significa que ela pode ser explciada
# pela combinacao da outras variaveis X´s
# ou seja, problema de multicolinearidade
# entao vamos calcular o VIF e Tolerancia

ols_vif_tol(stepw) #função ols_vif_tol do pacote olsrr
# ou seja, notamos que esse modelo pode oferecer uma 
# certa desconfianca sobre sua qualidade

#9d. Entendendo a situação do modelo Bodyfat stepwise criterio p-valor###############################

# no modelo que fizemos pelo stepwise criterio p-valor
summary(stepw_p)

# tinhamos essas variaveis no modelo
# observe que entre elas, existem altas correlacoes
vars_modelo_stepw_p <- Bodyfat %>% 
  select(Weight,Abdomen, Forearm, Wrist)
chart.Correlation(vars_modelo_stepw_p, histogram = TRUE)

# verificando VIF
ols_vif_tol(stepw_p)

