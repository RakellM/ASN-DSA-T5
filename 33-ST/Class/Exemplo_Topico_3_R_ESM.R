##Pacotes que precisam ser instalados
# install.packages("fpp2")
# install.packages("fpp3")
# install.packages("tidyverse")
# install.packages("tidyquant")
#install.packages("dplyr")


library(fpp2)         # An old forecasting framework
# library(fpp3)         # A new forecasting framework
library(tidyverse)    # Collection of data manipulation tools
library(tidyquant)    # Business Science ggplot theme

##########################################################################
#################### Exponential Smoothing Method ######################## 
##########################################################################
## Prós e Contras dos Modelos de Suavização Exponencial
## Prós
#  - Pode ser calculado rapidamente
#  - Generaliza bem para muitas séries temporais
## Contras
#  - Ignora a variação aleatória em favor de um processo de suavização
#  - Não pode ser usado para séries de natureza cíclica



#########################################################################################  
#########################################################################################  
#  Série - Quantidade remédio vendido para antidiabetes por mês de Jul/1991 - Jun/2008  #
#########################################################################################  
#########################################################################################  
# Base de dados 
drugs <- a10
plot(drugs)

##########################################################################
################ Methods without Trend and seasonality ################### 
##########################################################################
###### Simple Exponential Smoothing
# Fit and forecast with a SES model
# SES function
# ver mais detalhes em: https://www.rdocumentation.org/packages/forecast/versions/8.13/topics/ses
# ses(y, h = 10, level = c(80, 95), fan = FALSE, initial = c("optimal", "simple"),
#   alpha = NULL, lambda = NULL, biasadj = FALSE, x = y,  ...)

fc <- drugs %>% ses(h = 36) #posso definir o valor de alpha, usando , 
#alpha =0.8, ou deixar que o modelo encontre o melhor valor
#Para analisar os parâmetros do modelo
summary(fc)

#names(fc)
# [1] "model"     "mean"      "level"     "x"         "upper"     "lower"     "fitted"    "method"   
# [9] "series"    "residuals"
# fc$mean   # média (único componente desse tipo de modelo)
# fc$level  # intervalado de confiança 80 e 95% 


fc$method #mostra o método utilizado
fc$model  # modelo ajustado
#fc$fitted #mostra os valores ajustados

# Analisando os residuos para avaliar a qualidade do ajuste do modelo
plot(fc$residuals) 
# Check that the residuals look like white noise
checkresiduals(fc)
# p_valor < nível_significancia  => rejeitar a hipotese nula => série não é ruído branco

# Plot forecasts + one-step forecasts for the training data
autoplot(fc) + autolayer(fitted(fc))

##########################################################################




##########################################################################
######################### Methods with Trend ############################# 
##########################################################################

###########################
###### Holt's Linear Method
# Forecast using Holt's linear method
# ver mais detalhes em: https://www.rdocumentation.org/packages/forecast/versions/8.13/topics/ses
# holt( y,  h = 10, damped = FALSE,level = c(80, 95),fan = FALSE, initial = c("optimal", "simple"),
#   exponential = FALSE,alpha = NULL, beta = NULL,phi = NULL, lambda = NULL,biasadj = FALSE,x = y,  ...)
fcholt <- holt(drugs, h = 36)
#Para analisar os parâmetros do modelo
summary(fcholt)

names(fcholt)
fcholt$method
fcholt$model

# Analisando os residuos para avaliar a qualidade do ajuste do modelo
plot(fcholt$residuals) 

# Check if linear residuals look like white noise
checkresiduals(fcholt)
# p_valor < nível_significancia  => rejeitar a hipotese nula => série não é ruído branco

# Plot forecasts + one-step forecasts for the training data
autoplot(fcholt) + autolayer(fitted(fcholt))


###########################
###### Damped Method
# Forecast using damped method
fcdamped <- holt(drugs, damped = TRUE, h = 36)
#Para analisar os parâmetros do modelo
summary(fcdamped)


names(fcdamped)
fcdamped$method
fcdamped$model

# Analisando os residuos para avaliar a qualidade do ajuste do modelo
plot(fcdamped$residuals) 

# Check if damped residuals look like white noise
checkresiduals(fcdamped)
# p_valor < nível_significancia  => rejeitar a hipotese nula => série não é ruído branco

# Plot forecasts + one-step forecasts for the training data
autoplot(fcdamped) + autolayer(fitted(fcdamped))
##########################################################################




##########################################################################
################## Methods with Trend and seasonality #################### 
##########################################################################

#####################################
###### Holt-Winters' Additive  Method
# ver mais detalhes em: https://www.rdocumentation.org/packages/forecast/versions/8.13/topics/ses
# hw( y,h = 2 * frequency(x), seasonal = c("additive", "multiplicative"), damped = FALSE,
#   level = c(80, 95),fan = FALSE,initial = c("optimal", "simple"),  exponential = FALSE,
#   alpha = NULL,  beta = NULL,  gamma = NULL,  phi = NULL,  lambda = NULL,  biasadj = FALSE, x = y,  ...)
fc_hw_add <- hw(drugs, seasonal = "additive", h = 36)   # 3 years
#Para analisar os parâmetros do modelo
summary(fc_hw_add)


names(fc_hw_add)
fc_hw_add$method
fc_hw_add$model

# Analisando os residuos para avaliar a qualidade do ajuste do modelo
plot(fc_hw_add$residuals) 

# Check that the residuals look like white noise
checkresiduals(fc_hw_add)
# Plot original data + forecasts
autoplot(fc_hw_add) + autolayer(fitted(fc_hw_add))


#####################################
###### Holt-Winters' Multiplicative Method
# Forecast using Holt-Winters' multiplicative method
fc_hw_mult <- hw(drugs, seasonal = "multiplicative", h = 36)   # 3 years
#Para analisar os parâmetros do modelo
summary(fc_hw_mult)

names(fc_hw_mult)
fc_hw_mult$method
fc_hw_mult$model

# Analisando os residuos para avaliar a qualidade do ajuste do modelo
plot(fc_hw_mult$residuals) 

# Check that the residuals look like white noise
checkresiduals(fc_hw_mult)
# Plot original data + forecasts
autoplot(fc_hw_mult) + autolayer(fitted(fc_hw_mult))
##########################################################################


##########################################################################
####### Plotando todos os métodos anteriores no mesmo gráfico ############ 
##########################################################################
autoplot(drugs) +
autolayer(fitted(fc), series="Simples") +
autolayer(fitted(fcholt), series="Holt") +
autolayer(fitted(fcdamped), series="Damped") +
autolayer(fitted(fc_hw_add), series="Holt-Winters-Add") +
autolayer(fitted(fc_hw_mult), series="Holt-Winters-Multip") +
ggtitle("Venda de Remédio para antidiabetes") +
xlab("Mês/Ano") + ylab("Quantidade") +
guides(colour=guide_legend(title="Forecast"))






##########################################################################
########### Automatic Forecasting with Exponential Smoothing ############# 
##########################################################################

# Auto-fit an ETS model to airline data 
# ver mais detalhes em: https://www.rdocumentation.org/packages/forecast/versions/8.13/topics/ets
# ets(  y,  model = "ZZZ",  damped = NULL,  alpha = NULL,  beta = NULL,  gamma = NULL,  phi = NULL,
#   additive.only = FALSE,  lambda = NULL,  biasadj = FALSE,  lower = c(rep(1e-04, 3), 0.8),
#   upper = c(rep(0.9999, 3), 0.98),  opt.crit = c("lik", "amse", "mse", "sigma", "mae"),
#   nmse = 3,  bounds = c("both", "usual", "admissible"),  ic = c("aicc", "aic", "bic"),
#   restrict = TRUE,  allow.multiplicative.trend = FALSE,  use.initial.values = FALSE,
#   na.action = c("na.contiguous", "na.interp", "na.fail"),...)
fc_auto_no_seas <- ets(drugs)
#Para analisar os parâmetros do modelo
summary(fc_auto_no_seas)

names(fc_auto_no_seas)
fc_auto_no_seas$method
fc_auto_no_seas$par  


# Analisando os residuos para avaliar a qualidade do ajuste do modelo
plot(fc_auto_no_seas$residuals) 

# Check that the residuals look like white noise
checkresiduals(fc_auto_no_seas)

# Plot auto ETS forecasts
autoplot(forecast(fc_auto_no_seas, h = 36)) + autolayer(fitted(fc_auto_no_seas))


# Referências:
# https://medium.com/@JoonSF/moving-to-tidy-forecasting-in-r-an-overview-of-exponential-smoothing-methods-43794c9e2b8
# https://www.geeksforgeeks.org/exponential-smoothing-in-r-programming/
# Teoria: https://otexts.com/fpp3
