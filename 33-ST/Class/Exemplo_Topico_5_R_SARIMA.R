
#install.packages("forecast")
require(forecast)
#install.packages("urca")
library(urca)
#install.packages("tseries")
library(tseries)

library(lmtest)


######################################## 
######################################## 
### Série de nascimentos nos EUA 
######################################## 
######################################## 

######################################## 
# 1. Carregando a série:
birth = ts(read.table("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/birth.txt"))
length(birth)
plot(birth, main="Birth - Dados originais")
# Serie tem tendencia então não é estacionaria o que pode ser constatado também peelos graf acf e pacf abaixo

par(mfrow=c(2,1))
acf(birth,48,main="ACF: Birth - Dados originais")
pacf(birth,48,main="PACF: Birth - Dados originais")
par(mfrow=c(1,1))
# Pode-se notar que a série é não estacionária e possui um comportamento sazonal, pois a ACF decai lentamente 
# e tem aspecto de ondas, com picos nos lags 12, 24, 36 e 48. Pelo próprio gráfico da série original é possível
# notar este comportamento periódico

######################################## 
# 2. calculando a Primeira Diferença da série original
d1_birth = diff(birth)
plot(d1_birth,main="Birth - Primeira diferença")
#2.1 verificando se a primeria diff é estcionaria
adf.test(d1_birth)

######################################## 
# 3. Olhando ACF e PACF para definir os parametros
par(mfrow=c(2,1))
acf(d1_birth,48,main="ACF: Birth - Primeira diferença")
pacf(d1_birth,48,main="PACF: Birth - Primeira diferença")
par(mfrow=c(1,1))

######################################## 
# Diferença Sazonal, de ordem 12, da Primeira diferença do log da série original
d2_birth_12=diff(d1_birth,12)
plot(d2_birth_12,main="Birth - Diferença Sazonal, de ordem 12, da 1a dif")
par(mfrow=c(2,1))
acf(d2_birth_12,48,main="ACF: Birth - Diferença Sazonal, de ordem 12, da 1a dif")
pacf(d2_birth_12,48,main="PACF: Birth - Diferença Sazonal, de ordem 12, da 1a dif")
par(mfrow=c(1,1))

# podemos perceber que a PACF é significativamente diferente de zero até o lag 4, e nos lags 11 e 12. 
# A ACF é diferente de zero no lag 1 e no lag 12, principalmente, estando no limite em alguns outros lags. 
# Pelo comportamento da PACF no lag 11, sugere-se que há um componente MA sazonal de ordem 1. 
# Pelo truncamento da ACF no lag 4 (considerando os lags não sazonais, i.e., até o lag 12), 
# sugere-se um AR de ordem 4. 


######################################## 
# Vou testar alguns modelos
# 4. Estimação
fit1= arima(birth,order=c(4,1,0),seasonal=list(order=c(1,1,1),period=12))
fit2= arima(birth,order=c(4,1,0),seasonal=list(order=c(0,1,1),period=12))
fit3= arima(birth,order=c(4,1,1),seasonal=list(order=c(0,1,1),period=12))
fit4= arima(birth,order=c(4,1,1),seasonal=list(order=c(1,1,1),period=12))
# identificado em aula
fit5= arima(birth,order=c(4,1,2),seasonal=list(order=c(3,1,1),period=12))


fit1$aic
fit2$aic
fit3$aic
fit4$aic
fit5$aic


#Ajustando modelo com menor AIC
fit2= arima(birth,order=c(4,1,0),seasonal=list(order=c(0,1,1),period=12))
summary(fit2)
#library(lmtest)
coeftest(fit2) 


######################################## 
# 5. Diagnosticando o ajuste
#sarima(birth,4,1,0,0,1,1,12)
# usando teste Ljung-Box
tsdiag(fit2)


# Diagnosticando 
par(mfrow=c(1,5))
plot(fit2$resid,main="Resíduos")
acf(fit2$resid,main="ACF Resíduos",20)
pacf(fit2$resid,main="PACF Resíduos",20)
hist(fit2$resid,main="Histograma Resíduos")
qqnorm(fit2$resid,main="Normal Q-Q plot Resíduos")
par(mfrow=c(1,1))

# Ajustando uma previsão
previsao = forecast(fit2, h=12)
previsao
plot(previsao) 

#####################################################################################################################







######################################## 
######################################## 
### Série Johnson and Johnson (JJ)  
######################################## 
######################################## 

######################################## 
# 1. Carregando a série:
#  Série Original
jj = ts(read.table("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/jj.txt"))
length(jj)
plot(log(jj), main="JJ - Dados originais")
plot((jj), main="JJ - Dados originais")

par(mfrow=c(2,1))
acf(jj,48,main="ACF: JJ - Dados originais")
pacf(jj,48,main="PACF: JJ - Dados originais")
par(mfrow=c(1,1))

# Como a ACF da série decai lentamente de forma senoidal para dentro dos limites do intervalo de confiança e 
# a série apresenta tendência de crescimento, por isso aplicou-se a primeira diferença aos dados originais. 
# A Figura abaixo mostra o gráfico da primeira diferença da série original.

######################################## 
# 2. calculando a Primeira Diferença da série original
plot(diff(jj),main="JJ - Primeira diferença")

# a tendência foi removida, porém a variabilidade na primeira metade dos dados é bem menor que na 
# segunda metade. Dessa forma aplicou-se o logaritmo na série original e depois a primeira diferença. 
# Os resultados são apresentados nas Figuras 4 e 5 que exibem respectivamente a primeira diferença do 
# log da série original  e suas ACF e PACF.

######################################## 
# 3. Olhando ACF e PACF para definir os parametros
#   Primeira Diferença da série original
d1_l_jj = diff(log(jj))
plot(d1_l_jj,main="Log(JJ) - Primeira diferença")
par(mfrow=c(2,1))
acf(d1_l_jj,48,main="ACF: Log(JJ) - Primeira diferença")
pacf(d1_l_jj,48,main="PACF: Log(JJ) - Primeira diferença")
par(mfrow=c(1,1))

# os picos 4, 8, 12, 16, 20, ... tem decaimento relativamente lento sugerindo a presença de sazonalidade, 
# por isso a aplicação de uma diferença sazonal de ordem 4 é realizada abaixo

# Diferença Sazonal, de ordem , da Primeira diferença do log da série original
d2_l_jj_4=diff(d1_l_jj,4)
#d=1
#D=1
#s=4
plot(d2_l_jj_4,main="Log(JJ) - Diferença Sazonal, de ordem 4, da 1a dif")
par(mfrow=c(2,1))
acf(d2_l_jj_4,72,main="ACF: Log(JJ) - Diferença Sazonal, de ordem 4, da 1a dif")
pacf(d2_l_jj_4,72,main="PACF: Log(JJ) - Diferença Sazonal, de ordem 4, da 1a dif")
par(mfrow=c(1,1))


# Já vimos que d=1, pois foi aplicada a primeira diferença ao log da série desemprego, D=1 e s=4 
# pois aplicamos uma diferença sazonal de ordem 4 ao log da série JJ.
# Para obter os valore de P e Q basta olhar para os picos fora do intervalo de confiança nos lags 
# sazonais da PACF e da ACF respectivamente. Assim tomaremos Q=0 e P=1.
# Basta obtermos os valores de p e q para a parte não sazonal do modelo observando quantos picos 
# fora do intervalo de confiança ocorrem entre os lags sazonais na PACF e na ACF respectivamente. 
# Vamos ajustar alguns modelos SARIMA(p,1,q)X(1,1,0)4 para p=0,1,2 e q=0,1,2,3 e verificar os valores do AIC.

#SARIMA(p,d,q)(P,D,Q)s
#SARIMA(1,1,1)(1,1,0)4
#fit_inicial <- arima(log(jj), order = c(0,1,1),seasonal=list(order=c(1,1,0),period=4))
#summary(fit_inicial)
#coeftest(fit_inicial) 

######################################## 
#4. Definindo os valores dos parametros
# Vamos determinar os valores de p e q para a parte não sazonal do modelo
######ARIMA Models 
aic.table <- matrix(NA,nrow=3, ncol=3)
for (p in 0:2) for (q in 0:2) {
  aic.table[1+p,1+q] <- arima(log(jj), order = c(p,1,q),seasonal=list(order=c(1,1,0),period=4), method="ML")$aic
}
aic.table

#menor aic -150,9134 do modelo SARIMA(0,1,1)(1,1,0)4

#SARIMA(1,1,1)(1,1,0)4


######################################## 
# 5. Estimação
# Modelo escolhido pelo menor AIC
fit <- arima(log(jj), order = c(0,1,1),seasonal=list(order=c(1,1,0),period=4))#aic=-150,9134 ficar com este pelo menor numero de parametros
summary(fit)
#library(lmtest)
coeftest(fit) 


######################################## 
# 6. Diagnosticando o ajuste:
tsdiag(fit,gof.lag=50)

# Teste de normalidade dos resíduos.
#qqnorm(fit111$residuals)
shapiro.test(fit$residuals)


######################################## 
# 7. Ajustando uma previsão:
previsao = forecast(fit, h=12, level=c(90,95)); previsao
plot(previsao)

#####################################################################################################################
