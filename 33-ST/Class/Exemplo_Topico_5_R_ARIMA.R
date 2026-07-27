
#install.packages("forecast")
require(forecast)
#install.packages("urca")
library(urca)
#install.packages("tseries")
library(tseries)
library(lmtest)

#install.packages('astsa')
library(astsa)


###############################################################
# Como excluir coeficientes insignificantes com a função arima
# arima(x, order = c(2,0,5), fixed = c(NA,NA,NA,NA,0,0,0,NA))
# arima(x, order = c(2,0,5), fixed = c(0,NA,0,NA,NA,NA,NA,NA))
# p1
# p2
# d
# m1
# m2
# m3
# m4
# m5

################################################################################  
################################################################################  
############################## Série - Gas price ###############################
################################################################################  
################################################################################  

######################################## 
# 1. Carregando a série:
gas = ts(read.table("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/gas.txt"))

######################################## 
#2.1 Plotando a série original:
plot(gas,ylab="")

################################
#2.2 Verificando se a série é um Ruído Branco
#Usando Teste Ljung-Box
Box.test(gas, lag=10, type="Ljung-Box")
# p_valor < nível_significancia  => rejeitar a hipotese nula => série não é ruído branco


######################################## 
# 3. Verificando se a série é estacionária 
# O teste Augmented Dickey-Fuller (ADF) considera
# H0: Os dados não são estacionários 
# H1: Os dados são estacionários
# p_valor > nível_significancia  => aceitar a hipotese nula
# p_valor < nível_significancia  => rejeitar a hipotese nula

#Ver outros testes para detectar estacionariedade
#http://www.portalaction.com.br/series-temporais/14-testes-de-estacionariedade

#library(tseries)
adf.test(gas)
#Como p_valor > 0,05 => "aceita" a hipotese nula => a serie não é estacionaria


######################################## 
# 4. Diferenciando a série uma vez:
gas_diff = diff(gas,1)
plot.ts(gas_diff,type="o", xlab = "Anos", ylab="Primeira diferença da Série Preço do Petróleo")
#Verificando se a serie diferenciada é estacionária
adf.test(gas_diff)
# Podemos concluir que a seria diferenciada é estacionaria
# d=1

######################################## 
# 5. Definindo os valores de p e q
par(mfcol=c(1,2))
acf(gas_diff,main="")
pacf(gas_diff,main="")
par(mfcol=c(1,1))
#arima(2,1,1)

# removendo os parametros dem significancia e fiquei com o seguinte modelo
#fit <- arima(gas, order = c(0,1,1))
#summary(fit)
#library(lmtest)
#coeftest(fit)


# Decidindo o modelo pelo menor valor de AIC
maxp <- 2
maxq <- 1
#arima(0,1,0)
#arima(0,1,1)
#arima(1,1,0)
#arima(1,1,1)
#arima(2,1,0)
#arima(2,1,1)
aic.table <- matrix(NA,nrow=3, ncol=2)
bic.table <- matrix(NA,nrow=3, ncol=2)

n = 180
for (p in 0:2) for (q in 0:1) {
  aic.table[1+p,1+q] <- (log(arima(gas, order=c(p,1,q), method="ML")$sigma2) + (n + 2*(p+q+1))/n)
  bic.table[1+p,1+q] <- (log(arima(gas, order=c(p,1,q), method="ML")$sigma2) + (p+q+1)*log(n)/(n))
}

aic.table #menor aic é p=0 e q=1
bic.table #menor bic é p=0 e q=1

######################################## 
# 6. Ajustando e Diagnosticando o modelo escolhido 
# Ajustando
fit_011 <- arima(gas, order = c(0,1,1))
summary(fit_011)
#library(lmtest)
coeftest(fit_011) 

# usando teste Ljung-Box
tsdiag(fit_011)

# Diagnosticando 
par(mfrow=c(1,5))
plot(fit_011$resid,main="Resíduos")
acf(fit_011$resid,main="ACF Resíduos",20)
pacf(fit_011$resid,main="PACF Resíduos",20)
hist(fit_011$resid,main="Histograma Resíduos")
qqnorm(fit_011$resid,main="Normal Q-Q plot Resíduos")
par(mfrow=c(1,1))
# Teste de normalidade dos resíduos.
shapiro.test(fit_011$residuals)


######################################## 
# 7. Ajustando uma previsão e plotando os resultados
p <- predict(fit_011,10)
# Plotando os resultados do ajuste do modelo:
t1 = 1841:2020
t2 = 2021:2030
plot(t1,gas, type="o", xlim=c(1841, 2030), xlab = "Anos", ylab="Valores Observados + Previsão 10 anos")
lines(t2,p$pred, col="red", type="o")
lines(t2,p$pred - 1.96*p$se, col="blue", lty="dashed")
lines(t2,p$pred + 1.96*p$se, col="blue", lty="dashed")







################################################################################  
################################################################################  
########## Série - Globtemp2 - Temperatura Global anual de 1880-2004 ###########
################################################################################  
################################################################################

######################################## 
# 1. Carregando a série:
globtemp2 = ts(read.table("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/globtemp2.txt"))
globtemp2 = globtemp2[,2]

######################################## 
# 2. Plotando a série original:
t = 1880:2004
plot(t,globtemp2, type="o", xlab="Anos", ylab="Temperatura Global - globtemp2", )

######################################## 
# 3. Verificando se a série é estacionária 
adf.test(globtemp2)
#Como p_valor > 0,05 => "aceita" a hipotese nula => a serie não é estacionaria

######################################## 
# 4. Diferenciando a série uma vez:
globtemp2_diff = diff(globtemp2,1)
plot.ts(globtemp2_diff,type="o", xlab = "Anos", ylab="Primeira diferença na série Temperatuta Global")
#Verificando se a serie diferenciada é estacionária
adf.test(globtemp2_diff)
# Podemos concluir que a seria diferenciada é estacionaria

######################################## 
# 5. Definindo os valores de p e q
par(mfcol=c(1,2))
acf(globtemp2_diff,main="")
pacf(globtemp2_diff,main="")
par(mfcol=c(1,1))


# Decidindo o modelo pelo menor valor de AIC
maxp <- 5
maxq <- 4
aic.table <- matrix(NA,nrow=6, ncol=5)
bic.table <- matrix(NA,nrow=6, ncol=5)

n = 125
for (p in 0:5) for (q in 0:4) {
  aic.table[1+p,1+q] <- (log(arima(globtemp2, order=c(p,1,q), method="ML")$sigma2) + (n + 2*(p+q+1))/n)
  bic.table[1+p,1+q] <- (log(arima(globtemp2, order=c(p,1,q), method="ML")$sigma2) + (p+q+1)*log(n)/(n))
}

aic.table
bic.table

# Menor AIC é o modelo ARIMA(3,1,4)
# Menor BIC é o modelo ARIMA(0,1,2)

######################################## 
# 6. Ajustando e Diagnosticando o modelo escolhido 
# Ajustando
fit_012 <- arima(globtemp2, order = c(0,1,2)) 
summary(fit_012)
#library(lmtest)
coeftest(fit_012) 

# usando teste Ljung-Box
tsdiag(fit_012)


# Diagnosticando 
par(mfrow=c(1,5))
plot(fit_012$resid,main="Resíduos")
acf(fit_012$resid,main="ACF Resíduos",20)
pacf(fit_012$resid,main="PACF Resíduos",20)
hist(fit_012$resid,main="Histograma Resíduos")
qqnorm(fit_012$resid,main="Normal Q-Q plot Resíduos")
par(mfrow=c(1,1))
# Teste de normalidade dos resíduos.
shapiro.test(fit_012$residuals)



######################################## 
# 7. Ajustando uma previsão e plotando os resultados
p <- predict(fit_012,10)
# Plotando os resultados do ajuste do modelo:
t1 = 1880:2004
t2 = 2005:2014
plot(t1,globtemp2, type="o", ylim=c(-0.6,1), xlim=c(1880, 2020), xlab = "Anos", ylab="Valores Observados + Previsão 10 anos")
lines(t2,p$pred, col="red", type="o")
lines(t2,p$pred - 1.96*p$se, col="blue", lty="dashed")
lines(t2,p$pred + 1.96*p$se, col="blue", lty="dashed")




################################################################################  
################################################################################  
################################## AUTO ARIMA ##################################
################################################################################  
################################################################################



######################################## 
# Série - Globtemp2
######################################## 

globtemp2 = ts(read.table("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/globtemp2.txt"))
globtemp2 = globtemp2[,2]
plot(globtemp2,ylab="")


#Você pode escolher o modelo usando 
# aic, bic ou aicc
# e ajustar o modelo escolhido usando a função 
# auto.arima(x, stationary=FALSE, seasonal=TRUE,
#            ic=c("aicc","aic", "bic"), seasonal.test=c("ocsb","ch"),
#            allowdrift=TRUE, lambda=NULL, parallel=FALSE, num.cores=NULL)


##################################################################
#Usando AIC e estimando o modelo escolhido
##################################################################
modeloAIC = auto.arima(globtemp2,allowdrift = F, ic = 'aic')
summary(modeloAIC)
coeftest(modeloAIC) 
#ARIMA(3,1,2) 

tsdiag(modeloAIC) # avaliando o ajuste

previsao_aic = forecast(modeloAIC, h = 20) #previsao 4 periodos a frente
plot(previsao_aic) # plot previsão e intervalo de confiança


##################################################################
#Usando BIC e estimando o modelo escolhido
##################################################################
modeloBIC = auto.arima(globtemp2,allowdrift = F, ic = 'bic')
summary(modeloBIC)
coeftest(modeloBIC) 
#ARIMA(2,1,1)

tsdiag(modeloBIC) # avaliando o ajuste

previsao_bic = forecast(modeloBIC, h = 20) #previsao 4 periodos a frente
plot(previsao_bic) # plot previsão e intervalo de confiança


##################################################################
#Usando AIC e estimando o modelo escolhido
##################################################################
modeloAICc = auto.arima(globtemp2,allowdrift = F, ic = 'aicc')
summary(modeloAICc)
coeftest(modeloAICc) 
#ARIMA(3,1,2) 

tsdiag(modeloAICc) # avaliando o ajuste

previsao_aicc = forecast(modeloAICc, h = 4) #previsao 4 periodos a frente
plot(previsao_aicc) # plot previsão e intervalo de confiança









