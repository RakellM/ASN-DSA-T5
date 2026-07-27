#install.packages("forecast")
require(forecast)
#install.packages("urca")
library(urca)
#install.packages("tseries")
library(tseries)
#install.packages("lmtest")
library(lmtest)

 
# Dados mensais sobre temperatura global de 1880 a 2003
globtemp2 = ts(read.table("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/globtemp2.txt"))
globtemp2 = diff(globtemp2[,2])

################################
#1. Visualizando a série temporal
t = 1880:2003
plot(t,globtemp2, type="o", xlab="Anos", ylab="Temperatura Global - globtemp2", )


################################
#2. Verificando se a série é um Ruído Branco
#Usando Teste Ljung-Box
Box.test(globtemp2, lag=10, type="Ljung-Box")
# p_valor < nível_significancia  => rejeitar a hipotese nula => série não é ruído branco

################################
#3. Verificando se a série é Estacionária
# ADF test 
adf.test(globtemp2)
# p_valor < nível_significancia  => rejeita a hipotese nula => série é estacionaria


################################
#4. IDENTIFICAÇÃO
# Verificando autocorrelação e autocorrelação parcial
par(mfcol=c(1,2))
acf(globtemp2,main="")
pacf(globtemp2,main="")
par(mfcol=c(1,1))
# ggAcf(mort,48)
# ggPacf(mort,48)



################################
# 5. ESTIMAÇÃO
# Ajusta o modelo
# arima(p,d,q)
fit_arma <- arima(globtemp2, order = c(5,0,4)) 
summary(fit_arma)

#library(lmtest)
coeftest(fit_arma) 
#aic = -146.54 

# ajuste da aula
#fit_arma_1_test <- arima(globtemp2, order = c(3,0,2), fixed = c(NA,0,NA,0,NA,NA)) 
#coeftest(fit_arma_1_test) 
#tsdiag(fit_arma_1_test)
#plot(forecast(fit_arma_1_test, h=15, level=c(80,85,90,95)))

# remover os parametros da parte MA, tem p_valores muito alto
fit_arma_1 <- arima(globtemp2, order = c(5,0,4), fixed = c(NA,NA,NA,0,NA,0,NA,NA,0,NA)) 

summary(fit_arma_1)
coeftest(fit_arma_1) 
#aic = -148.6

# remover os parametros da parte AR:4 e 5, tem p_valores não significantes
fit_arma_2 <- arima(globtemp2, order = c(3,0,0)) 
summary(fit_arma_2)
coeftest(fit_arma_2) 
#aic = -145.91
# modelo ficou pior então vou seguir com o fit_arma_1

################################
# 6. VERIFICAÇÃO
# Avalia a qualidade do ajuste 

# Diagnosticando com bases nas definições de ruido branco
par(mfrow=c(1,5))
plot(fit_arma$resid,main="Resíduos")
acf(fit_arma$resid,main="ACF Resíduos",20)
pacf(fit_arma$resid,main="PACF Resíduos",20)
hist(fit_arma$resid,main="Histograma Resíduos")
qqnorm(fit_arma$resid,main="Normal Q-Q plot Resíduos")
par(mfrow=c(1,1))

# usando teste Ljung-Box
tsdiag(fit_arma_1)
# Teste de normalidade dos resíduos.
#shapiro.test(fit_ar_1$residuals)



################################
# 6. PREVISÃO
previsao <- forecast(fit_arma) 
plot(forecast(fit_arma, h=15, level=c(80,85,90,95)))


