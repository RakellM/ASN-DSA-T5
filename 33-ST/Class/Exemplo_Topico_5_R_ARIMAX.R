
#install.packages("forecast")
require(forecast)
#install.packages("urca")
library(urca)
#install.packages("tseries")
library(tseries)
library(ggplot2)
#install.packages("TSA")
library(TSA)
library(MASS)
#library(lmtest)

#install.packages("Ecdat")
library(Ecdat)
library(lmtest)
 
#install.packages("gridExtra")
require(gridExtra)


# Icecream data set 
# variaveis disponíveis
    # cons: ice cream consumption in the USA (in pints, per capita),
    # income: average family income per week (in USD),
    # price: price of ice cream (per pint), and
    # temp: average temperature (in Fahrenheit).
# The number of observations is 30. 
# They correspond to four-weekly periods in the span from March 18, 1951 to July 11, 1953.

# data(Icecream) # from package Ecdat
# ou baixe o arquivo em https://www.r-exercises.com/wp-content/uploads/2017/04/Icecream.csv



##Part 1
#Load the dataset, and plot the variables cons (ice cream consumption), temp (temperature), and income
df <- read.csv("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/Icecream.csv")
head(df)
p1 <- ggplot(df, aes(x = X, y = cons)) +
  ylab("Consumption") +
  xlab("") +
  geom_line() +
  expand_limits(x = 0, y = 0)
p2 <- ggplot(df, aes(x = X, y = temp)) +
  ylab("Temperature") +
  xlab("") +
  geom_line() +
  expand_limits(x = 0, y = 0)
p3 <- ggplot(df, aes(x = X, y = income)) +
  ylab("Income") +
  xlab("Period") +
  geom_line() +
  expand_limits(x = 0, y = 0)
grid.arrange(p1, p2, p3, ncol=1, nrow=3)


#Part 2
# Estimate an ARIMA model for the data on ice cream consumption using the auto.arima function. 
# Then pass the model as input to the forecast function to get a forecast for the next 6 periods 
# (both functions are from the forecast package).
#require(forecast)
fit_cons <- auto.arima(df$cons)
summary(fit_cons)
coeftest(fit_cons) 
#ARIMA(3,0,0) with non-zero mean 

fcast_cons <- forecast(fit_cons, h = 6)
# 
# #Part 3
# Plot the obtained forecast with the autoplot.forecast function from the forecast package.
#require(forecast)
autoplot(fcast_cons)
# 
# #Part 4
# Use the accuracy function from the forecast package to find the mean absolute scaled error (MASE) 
# of the fitted ARIMA model.
accuracy(fit_cons)
## The MASE is equal to 0.8200619
# 
# #Part 5
# Estimate an extended ARIMA model for the consumption data with the temperature variable as 
# an additional regressor (using the auto.arima function). Then make a forecast for the next 
# 6 periods (note that this forecast requires an assumption about the expected temperature; 
# assume that the temperature for the next 6 periods will be represented by the following vector: 
# fcast_temp <- c(70.5, 66, 60.5, 45.5, 36, 28)).
# Plot the obtained forecast.
fit_cons_temp <- auto.arima(df$cons, xreg = df$temp)
summary(fit_cons_temp)
coeftest(fit_cons_temp)

fcast_temp <- c(70.5, 66, 60.5, 45.5, 36, 28)
fcast_cons_temp <- forecast(fit_cons_temp, xreg = fcast_temp, h = 6)
autoplot(fcast_cons_temp)
accuracy(fit_cons_temp)
# 
# 
# #Part 6
# Print summary of the obtained forecast. Find the coefficient for the temperature variable, 
# its standard error, and the MASE of the forecast. Compare the MASE with the one of the initial forecast.
summary(fcast_cons_temp)
# the coefficient for the temperature variable is 0.0028
# the standard error of the coefficient is 0.0007
# the mean absolute scaled error is 0.7354048, which is smaller than
# the error for the initial model (0.8200619)


# 
# #Part 7
# Check the statistical significance of the temperature variable coefficient using the 
# coeftest function from the lmtest package. Is the coefficient statistically significant at 5% level?
#require(lmtest)
#coeftest(fit_cons_temp)

#   
# #Part 8
####################################
# Função de Correlação CRuzada - PLOT
#verificando correlação da var consumption com a var temperatura
ccf(df$cons, df$temp) 

#verificando correlação da var consumption com a var temperatura
ccf(df$cons, df$income)

ccf(df$cons, df$price)

# The function that estimates the ARIMA model can input more additional regressors, 
# but only in the form of a matrix. Create a matrix with the following columns:
#   values of the temperature variable,
#   values of the income variable,
#   values of the income variable lagged one period,
#   values of the income variable lagged two periods.
# Print the matrix.
# Note: the last three columns can be created by prepending two NA's to the vector of values of the 
# income variable, and using the obtained vector as an input to the embed function (with the dimension 
# parameter equal to the number of columns to be created).
temp_column <- matrix(df$temp, ncol = 1)
income <- c(NA, NA, df$income)
income_matrix <- embed(income, 3)
vars_matrix <- cbind(temp_column, income_matrix)
print(vars_matrix)
#Ver mais exemplos em: https://nwfsc-timeseries.github.io/atsa-labs/sec-tslab-correlation-within-and-among-time-series.html

# 
# #Part 9
# Use the obtained matrix to fit three extended ARIMA models that use the following variables as 
# additional regressors:
#   temperature, income,
#   temperature, income at lags 0, 1,
#   temperature, income at lags 0, 1, 2.
# Examine the summary for each model, and find the model with the lowest value of the Akaike information 
# criterion (AIC).
# Note that the AIC cannot be used for comparison of ARIMA models with different orders of integration 
# (expressed by the middle terms in the model specifications) because of a difference in the number of observations. 
# For example, an AIC value from a non-differenced model, ARIMA (p, 0, q), cannot be compared to the 
# corresponding value of a differenced model, ARIMA (p, 1, q).
fit_vars_0 <- auto.arima(df$cons, xreg = vars_matrix[, 1:2])
summary(fit_vars_0)
coeftest(fit_vars_0)
# Regression with ARIMA(1,0,0) errors + xreg1 + xreg2
fit_vars_1 <- auto.arima(df$cons, xreg = vars_matrix[, 1:3])
summary(fit_vars_1)
coeftest(fit_vars_1)

fit_vars_2 <- auto.arima(df$cons, xreg = vars_matrix[, 1:4])
summary(fit_vars_2)
coeftest(fit_vars_2)

print(fit_vars_0$aic)
print(fit_vars_1$aic)
print(fit_vars_2$aic)
# The AIC can be used because the models have the same order of integration (0).
# The model with the lowest value of the AIC is the first model.
# Its AIC is equal to -113.3357.



# 
# #Part 10
# Use the model found in the previous exercise to make a forecast for the next 6 periods, and plot the forecast. 
# (The forecast requires a matrix of the expected temperature and income for the next 6 periods; 
#   create the matrix using the fcast_temp variable, and the following values 
#   for expected income: 91, 91, 93, 96, 96, 96).
# Find the mean absolute scaled error of the model, and compare it with the ones from the first two models 
# in this exercise set.
expected_temp_income <- matrix(c(fcast_temp, 91, 91, 93, 96, 96, 96),
                               ncol = 2, nrow = 6)
fcast_cons_temp_income <- forecast(fit_vars_0,
                                   xreg = expected_temp_income,
                                   h = 6)
autoplot(fcast_cons_temp_income)

accuracy(fit_cons)[, "MASE"]
## [1] 0.7542003
accuracy(fit_cons_temp)[, "MASE"]
## [1] 0.7354048
accuracy(fit_vars_0)[, "MASE"]
## [1] 0.7290753

# the model with two external regressors has the lowest 
# mean absolute scaled error (0.7290753)





