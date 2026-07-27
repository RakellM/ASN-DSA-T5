#install.packages("forecast")
require(forecast)
#install.packages("urca")
library(urca)
#install.packages("tseries")
library(tseries)
#install.packages("lmtest")
library(lmtest)


################### Exercicio 3.17  
# Dados diários sobre mortalidade cardiovascular
mort = ts(read.table("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/cmort.txt"))


################################
#1. Visualizando a série temporal
plot(mort)


################################
#2. Verificando se a série é um Ruído Branco
#Usando Teste Ljung-Box
Box.test(mort, lag=10, type="Ljung-Box")
# p_valor < nível_significancia  => rejeitar a hipotese nula => série não é ruído branco

################################
#3. Verificando se a série é Estacionária
# ADF test 
adf.test(mort)
# p_valor < nível_significancia  => rejeita a hipotese nula => série é estacionaria


################################
#4. IDENTIFICAÇÃO
# Verificando autocorrelação e autocorrelação parcial
par(mfcol=c(1,2))
acf(mort,main="")
pacf(mort,main="")
par(mfcol=c(1,1))
# ggAcf(mort,48)
# ggPacf(mort,48)



################################
# 5. ESTIMAÇÃO
# Ajusta o modelo
# arima(p,d,q)
#p -> AR
#q -> MA
#d -> diferenciação
fit_ar_2 <- arima(mort, order = c(2,0,0)) #arima(2,0,0)=AR(2)
summary(fit_ar_2)

coeftest(fit_ar_2) 


################################
# 6. VERIFICAÇÃO
# Avalia a qualidade do ajuste 
tsdiag(fit_ar_2)
# Teste de normalidade dos resíduos.
#shapiro.test(fit_ar_1$residuals)

# ou
Box.test(fit_ar_2$residuals, lag=10, type="Ljung-Box")
# p_valor > nível_significancia  => não rejeitar a hipotese nula => "série" é ruído branco




################################
# 7. PREVISÃO
previsao <- forecast(fit_ar_2) 
plot(forecast(fit_ar_2, h=15, level=c(80,85,90,95)))


#MA(4)
#arima(p,d,q)
#fit_ma_4 <- arima(mort, order = c(0,0,4))
