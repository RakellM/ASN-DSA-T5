

library(plotly)
#install.packages("fpp")
library(fpp)
library(ggplot2)

#install.packages("hts")
library(hts)

#install.packages("zoo")
library(zoo)



####################################################
####################################################
################ 1. hts vn dataset #################
####################################################
####################################################
# Referencia para esse exemplo: 
# https://medium.com/brillio-data-science/forecasting-hierarchical-time-series-using-r-598828dba435

# Data set
# Base de dados trimestral com o total de dias hospedados, de 1998-2011, 
# para visitantes de oito regiões da Austrália:
data(vn) # vem do pacote fpp
plot(vn)
head(vn)
vn
summary(vn)

# criando a hierarquia temporal
y <- hts(vn, nodes=list(4,c(2,2,2,2)))
y
# The above command creates a hierarchical time series with 3 levels(top most level one does 
# not have to specify) with 4 nodes/states in the middle and 8 nodes/cities in bottom most level.
# (Argument 'nodes' does the trick for you here,also notice 2 cities are tagged to each state.)

# Split in train and holdout data
data <- window(y, start = 1998, end = 2009)
test <- window(y, start = 2010)

# Documentation for hts package
#https://cran.r-project.org/web/packages/hts/hts.pdf


#Syntax
# forecast(
#   object,
#   h = ifelse(frequency(object$bts) > 1L, 2L * frequency(object$bts), 10L),
#   method = c("comb", "bu", "mo", "tdgsa", "tdgsf", "tdfp"),
#   weights = c("wls", "ols", "mint", "nseries"),
#   fmethod = c("ets", "arima", "rw"),
#   algorithms = c("lu", "cg", "chol", "recursive", "slm"),
#   covariance = c("shr", "sam"),
#   nonnegative = FALSE,
#   control.nn = list(),
#   keep.fitted = FALSE,
#   keep.resid = FALSE,
#   positive = FALSE,
#   lambda = NULL,
#   level,
#   FUN = NULL,
#   xreg = NULL,
#   newxreg = NULL,
#   parallel = FALSE,
#   num.cores = 2,
#   ...
# )


#1. ETS
#method = tdfp (top-down forecast proportions)
#fmethod = ets (exponential smoothing)
ETS_Top_Down <- forecast(y, h=8,method = 'tdfp',fmethod = 'ets')
plot(ETS_Top_Down)
names(ETS_Top_Down)
accuracy(ETS_Top_Down, test)

ETS_Bottom_Up <- forecast(y, h=8,method = 'bu',fmethod = 'ets')
plot(ETS_Bottom_Up)
accuracy(ETS_Bottom_Up, test)

ETS_Middle_Out <- forecast(y, h=8,method = 'mo',fmethod = 'ets',level=2)
plot(ETS_Middle_Out)
accuracy(ETS_Middle_Out, test)

# Adjusting model in train data and verify accuracy in test (holdout) data
fcasts_ets <- forecast(data,h=8,method = 'mo',fmethod = 'ets',level=2)
accuracy(fcasts_ets, test)


#2. ARIMA
#method = tdfp (top-down forecast proportions)
#fmethod = ets (exponential smoothing)
ARIMA_Top_Down <- forecast(y, h=8,method = 'tdfp',fmethod = 'arima')
plot(ARIMA_Top_Down)

ARIMA_Bottom_Up <- forecast(y, h=8,method = 'bu',fmethod = 'arima')
plot(ARIMA_Bottom_Up)

ARIMA_Middle_Out <- forecast(y, h=8,method = 'mo',fmethod = 'arima',level=2)
plot(ARIMA_Middle_Out)
# Adjusting model in train data and verify accuracy in test (holdout) data
fcasts_arima <- forecast(data,h=8,method = 'mo',fmethod = 'arima',level=2)
accuracy(fcasts_arima, test)


#3. RW - Randow Walk
#method = tdfp (top-down forecast proportions)
#fmethod = ets (exponential smoothing)
RW_Top_Down <- forecast(y, h=8,method = 'tdfp',fmethod = 'rw')
plot(RW_Top_Down)

RW_Bottom_Up <- forecast(y, h=8,method = 'bu',fmethod = 'rw')
plot(RW_Bottom_Up)

RW_Middle_Out <- forecast(y, h=8,method = 'mo',fmethod = 'rw',level=2)
plot(RW_Middle_Out)
# Adjusting model in train data and verify accuracy in test (holdout) data
fcasts_rw <- forecast(data,h=8,method = 'mo',fmethod = 'rw',level=2)
accuracy(fcasts_rw, test)

names(fcasts_rw)



