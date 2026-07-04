
#install.packages("fpp2")
library(fpp2)         # An old forecasting framework
# library(fpp3)         # A new forecasting framework
library(tidyverse)    # Collection of data manipulation tools

#install.packages("tidyquant")
library(tidyquant)    # Business Science ggplot theme

#install.packages("urca")
library(urca)
library(tseries)

# Base de dados 
# Quantidade remédio vendido para antidiabetes por mês de Jul/1991 - Jun/2008
# base com aplicação de log
drugs <- a10
head(drugs,50)

# Verificando que a base é um objeto serie temporal
class(drugs)
# se não for um objeto de serie temporal você pode utilizar o comando ts(nome_base) para transformá-lo em uma serie temporal



################################
################################
#1. Visualizando a série temporal
plot(drugs)

################################
################################
#2. Verificando se há presença de sazonalidade
# Opção 2.1: Dados plotados sobre o mesmo eixo
ggseasonplot(drugs, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("$ million") +
  ggtitle("Seasonal plot: antidiabetic drug sales")

# Opção 2.2: Dados plotados com coordenadas polares
ggseasonplot(drugs, polar=TRUE) +
  ylab("$ million") +
  ggtitle("Polar seasonal plot: antidiabetic drug sales")


################################
################################
#3. Fazendo a decomposição dos elementos da série
# Para mais detalhes ver em: https://otexts.com/fpp2/classical-decomposition.html

# 3.1 Usando decomposição multiplicativa
# yt = St x Tt x Rt
# que é equivalente a log(yt) = log(St) + log(Tt) + log(Rt)
drugs %>% decompose(type="multiplicative") %>%
  autoplot() + xlab("Year") +
  ggtitle("Decomposição classica Multiplicativa para os dados Drugs")

# 3.1 Usando decomposição aditiva
# yt = St + Tt+ Rt
drugs %>% decompose(type="additive") %>%
  autoplot() + xlab("Year") +
  ggtitle("Decomposição classica Aditiva para os dados Drugs")


################################
################################
#4. Verificando autocorrelação
# Usando função ACF 
ggAcf(drugs,48)




################################
################################
#5. Verificando se a série é um Ruído Branco

#5.1 Usando Teste Ljung-Box
Box.test(drugs, lag=10, type="Ljung-Box")
# p_valor < nível_significancia  => rejeitar a hipotese nula => série não é ruído branco


# #5.2 Usando as definições do que é ser um Ruído Branco
# #### Usando funções ggplot2
# # Verificando se há variáção sistemática
# autoplot(drugs) + xlab("Mês") + ylab("") + ggtitle("Série Drugs")
# #Olhando se a distribuição é normal
# gghistogram(drugs) + ggtitle("Histograma da Série Drugs")
# # Verificando se há correlação com as observações passadas
# ggAcf(drugs) + ggtitle("ACF da Série Drugs")

#### Usando outras funções
par(mfrow=c(1,4))
plot(drugs,main="Serie")
acf(drugs,main="ACF",24)
hist(drugs,main="Histograma")
qqnorm(drugs,main="Normal Q-Q plot")
par(mfrow=c(1,1))


################################
################################
#6. Verificando se a série é Estacionária

#library(tseries)
# ADF test 
#ADF test, where the null hypothesis is the time series possesses a unit root and is non-stationary. 
#So, id the P-Value in ADH test is less than the significance level (0.05), you reject the null hypothesis.
#H0:  A série não estacionária.
#H1:  A série é estacionária
adf.test(drugs, k=10)
# p_valor > nível_significancia  => aceitar a hipotese nula => série não é estacionaria
#a10_txt <- read.table("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/a10_serie.txt")


#library(tseries)
# KPSS teste
#The KPSS test, on the other hand, is used to test for trend stationarity. 
#The null hypothesis and the P-Value interpretation is just the opposite of ADF test.
#H0:  A série estacionária.
#H1:  A série não é estacionária

kpss.test(drugs)
# p_valor <  nível_significancia  => rejeitar a hipotese nula => série não é estacionaria


