require(forecast)
library(urca)
library(tseries)
library(fpp2)         # An old forecasting framework
library(tidyverse)    # Collection of data manipulation tools
library(tidyquant)    # Business Science ggplot theme



#########################################################################################  
#########################################################################################  
#  Série - Quantidade remédio vendido para antidiabetes por mês de Jul/1991 - Jun/2008  #
#########################################################################################  
#########################################################################################  


######################################## 
# 1. Carregando e plotando a a série:
drugs <- a10
head(drugs,10)

autoplot(drugs) + xlab("Mês/Ano") + ylab("GWh") +
  ggtitle("Venda de Remédio para antidiabetes")



######################################## 
# 2. Modelo "Média Móvel" 
# Com 3 periodos de defasagem
#ma(drugs, 3)
autoplot(drugs, series="Data") +
  autolayer(ma(drugs,3), series="3-MA") +
  xlab("Mês/Ano") + ylab("Quantidade") +
  ggtitle("Venda de Remédio para antidiabetes") +
  scale_colour_manual(values=c("Data"="grey50","3-MA"="red"),
                      breaks=c("Data","3-MA"))

# Com 4 periodos de defasagem
#ma(drugs, 4)
autoplot(drugs, series="Data") +
  autolayer(ma(drugs,4), series="4-MA") +
  xlab("Mês/Ano") + ylab("Quantidade") +
  ggtitle("Venda de Remédio para antidiabetes") +
  scale_colour_manual(values=c("Data"="grey50","4-MA"="red"),
                      breaks=c("Data","4-MA"))
# Com 5 periodos de defasagem
#ma(drugs, 5)
autoplot(drugs, series="Data") +
  autolayer(ma(drugs,5), series="5-MA") +
  xlab("Mês/Ano") + ylab("Quantidade") +
  ggtitle("Venda de Remédio para antidiabetes") +
  scale_colour_manual(values=c("Data"="grey50","5-MA"="red"),
                      breaks=c("Data","5-MA"))
# Com 7 periodos de defasagem
#ma(drugs, 7)
autoplot(drugs, series="Data") +
  autolayer(ma(drugs,7), series="7-MA") +
  xlab("Mês/Ano") + ylab("Quantidade") +
  ggtitle("Venda de Remédio para antidiabetes") +
  scale_colour_manual(values=c("Data"="grey50","7-MA"="red"),
                      breaks=c("Data","7-MA"))

# Plotando todos os gráficos juntos
autoplot(drugs) +
  autolayer(ma(drugs,3),
            series="3-MA", PI=FALSE) +
  autolayer(ma(drugs,4),
            series="4-MA", PI=FALSE) +
  autolayer(ma(drugs,5),
            series="5-MA", PI=FALSE) +
  autolayer(ma(drugs,7),
            series="7-MA", PI=FALSE) +
  ggtitle("Venda de Remédio para antidiabetes") +
  xlab("Mês/Ano") + ylab("Quantidade") +
  guides(colour=guide_legend(title="Forecast"))




######################################## 
# 4. Modelo Average 
mean_drugs <- meanf(drugs, 5)
summary(mean_drugs)

######################################## 
# 5. Modelo Naive 
naive_drugs <- naive(drugs, 3)
summary(naive_drugs)
rwf(drugs, 3) # Equivalent alternative
autoplot(drugs) +
  autolayer(naive(drugs, 12),
            series="Naïve", PI=FALSE)

######################################## 
# 6. Modelo Naive Sazonal
snaive_drugs <- snaive(drugs, 12)
summary(snaive_drugs)
autoplot(drugs) +
  autolayer(snaive(drugs, h=24),
            series="Seasonal naïve", PI=FALSE)

######################################## 
# 7. Modelo Drift
rwf_drugs <- rwf(drugs, 12,drift=TRUE)
summary(rwf_drugs)
autoplot(drugs) +
  autolayer(rwf(drugs, 12,drift=TRUE),
            series="Drift", PI=FALSE)

#################################
# Plotando todos os gráficos juntos
autoplot(drugs) +
  autolayer(meanf(drugs, h=24),
            series="Mean", PI=FALSE) +
  autolayer(naive(drugs,3, h=24),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(drugs, h=24),
            series="Seasonal naïve", PI=FALSE) +
  autolayer(rwf(drugs,drift=TRUE, h=24),
            series="Drift", PI=FALSE) +
  ggtitle("Venda de Remédio para antidiabetes") +
  xlab("Mês/Ano") + ylab("Quantidade") +
  guides(colour=guide_legend(title="Forecast"))







