# MULTIPLE LINEAR REGRESSION
##########################################################################

# %%
## LIBRARY
##################################
import pandas as pd #- data manipulation
import os #- system
import statsmodels.api as sm #- logistic regression
import numpy as np #- mathematical calculations
from matplotlib import pyplot as plt #- data visualization 
import seaborn as sns #- data visualization 
from tabulate import tabulate # pretty tables
from sklearn.metrics import mean_squared_error, mean_absolute_error #- error function
from statsmodels.stats.outliers_influence import variance_inflation_factor #- VIF
import scipy.stats as stats #- data modeling
import pylab #- QQplot
from statsmodels.stats.diagnostic import het_breuschpagan #- Breusch-Pagan variance test
from scipy.stats import shapiro #- Shapiro-Wilk normality test

import kagglehub #- Kaggle Hub to import datasets

# %%
### Discussion: Selection mothods Backward, Forward and Stepwise
#- Python does not has the selection methods built in on the most used packages.
#- Solution here is to create these functions.
#- Check file: linear_regression_functions.py
# from linear_regression_functions import step, charts_var_num, plot_regression_diagnostics, hist_bins=10, figsize=(10, 6), 
from linear_regression_functions import *

# %% 
## DATASET
##################################
#- Dataset for multiple regression
#- KaggleHub
path = kagglehub.dataset_download(r'ruchikakumbhar/calories-burnt-prediction')
df = pd.read_csv(os.path.join(path, 'calories.csv'))  
print(df.head())

# %%
## EDA
##################################
### Summary of Dependent Variable (Y)
df["Calories"].describe()

# %%
#- Histogram
sns.histplot(data=df, x="Calories")

# %%
#- Box-Plot
sns.boxplot(df["Calories"])

# %%
#- Scatter Plot 
#- 1. Duration vs Calories
df.plot(kind = "scatter", x= "Duration", y="Calories")

# %%
#- Scatter Plot 
#- 2. Weight vs Calories
df.plot(kind = "scatter", x= "Body_Temp", y="Calories")

# %%
### Analyse Multicolinearity
df_numeric = df.drop(["Gender", "User_ID"], axis=1)

#### Correlation Matriz 
#- R style
python_chart_correlation(df_numeric, 
                         hist_bins=10, 
                         figsize=(10, 6), 
                         fontsize=12, 
                         color='red', 
                         scatter_alpha=0.6)

# %%
#- Plot correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# %%
df_model = df.drop(["User_ID", 
                    "Gender",
                    "Heart_Rate",
                    "Body_Temp",
                    "Height"
                    ], axis=1)

# %%
## MODEL
##################################
# Independent Variables (Matrix)
X = df_model.drop(["Calories"] , axis=1)
# Dependent Variable (vector)
y = df_model["Calories"] 
# Add the constant column on the X matrix
X = sm.add_constant(X) 
# Adjusting the model
model = sm.OLS(y, X).fit()

# Summary
print(model.summary())

# %%
## RESIDUAL ANALYSIS
##################################
fig = plot_regression_diagnostics(model)
plt.show()

# %%
## MODEL FULL
##################################
#- Independent Variables (Matrix)
X = df.drop(["Calories","User_ID", "Gender"] , axis=1)
#- Dependent Variable (vector)
y = df["Calories"] 
#- Add the constant column on the X matrix
X = sm.add_constant(X) 
#- Adjusting the model
model_full = sm.OLS(y, X).fit()
#- Summary
print(model_full.summary()) 
#- Generate predicted
pred_full = model_full.predict(X)


# %%
## MODEL Method Forward
##################################
### AIC
#- Find variables X relevants to the model

#- Independent Variables (Matrix)
X = df.drop(["Calories", "User_ID", "Gender"] , axis=1)
#- Dependent Variable (vector)
y = df["Calories"]

#- List of variables
columns_forw = step(var_dependent = 'Calories', 
                    var_independent = X.columns.to_list(), 
                    dataset = df, 
                    method = 'forward', 
                    metric = 'aic') 
columns_forw

# %%
#- Running the model

#- Get the variable list
X_forw = df[columns_forw['var'].to_list()[0] ]
#- Add the intercept
X_forw = sm.add_constant(X_forw)
#- Run the model
forw = sm.OLS(y, X_forw).fit()
#- Summary
print(forw.summary()) 
#- Generate predicted
pred_forw = forw.predict(X_forw)

# %%
### BIC
#- List of variables
columns_forw = step(var_dependent = 'Calories', 
                    var_independent = X.columns.to_list(), 
                    dataset = df, 
                    method = 'forward', 
                    metric = 'bic') 
columns_forw

# %%
### p-value
#- List of variables
columns_forw = step(var_dependent = 'Calories', 
                    var_independent = X.columns.to_list(), 
                    dataset = df, 
                    method = 'forward', 
                    metric = 'pvalue') 
columns_forw

# %%
## MODEL Method Backward
##################################
### AIC
#- Independent Variables (Matrix)
X = df.drop(["Calories", "User_ID", "Gender"] , axis=1)
#- Dependent Variable (vector)
y = df["Calories"]

#- List of variables
columns_back = step(var_dependent = 'Calories', 
                     var_independent = X.columns.to_list(), 
                     dataset = df, 
                     method = 'backward' ,
                     metric = 'aic')
columns_back

# %%
#- Running the model

#- Get the variable list
X_backw = df [ columns_back['var'].to_list()[0] ] 
#- Add the intercept
X_backw = sm.add_constant(X_backw)
#- Run the model
backw = sm.OLS(y, X_backw).fit()
#- Summary
print(backw.summary()) 
#- Generate predicted
pred_backw = backw.predict(X_backw)

# %%
### BIC
#- List of variables
columns_back = step(var_dependent = 'Calories', 
                     var_independent = X.columns.to_list(), 
                     dataset = df, 
                     method = 'backward' ,
                     metric = 'bic')
columns_back

# %%
### p-value
#- List of variables
columns_back = step(var_dependent = 'Calories', 
                     var_independent = X.columns.to_list(), 
                     dataset = df, 
                     method = 'backward' ,
                     metric = 'pvalue')
columns_back

# %%
## MODEL Stepwise
##################################
### AIC
#- Independent Variables (Matrix)
X = df.drop(["Calories", "User_ID", "Gender"] , axis=1)
#- Dependent Variable (vector)
y = df["Calories"]

#- List of variables
columns_stepw = step(var_dependent = 'Calories', 
                     var_independent = X.columns.to_list(), 
                     dataset = df,
                     method = 'both' ,
                     metric = 'aic')
columns_stepw

# %%
#- Running the model

#- Get the variable list
X_stepw = df [ columns_stepw['var'].to_list()[0] ] 
#- Add the intercept
X_stepw = sm.add_constant(X_stepw)
#- Run the model
stepw = sm.OLS(y, X_stepw).fit()
#- Summary
print(stepw.summary()) 
#- Generate predicted
pred_stepw = stepw.predict(X_stepw)

# %%
### BIC
#- List of variables
columns_stepw = step(var_dependent = 'Calories', 
                     var_independent = X.columns.to_list(), 
                     dataset = df,
                     method = 'both' ,
                     metric = 'bic')
columns_stepw

# %%
### p-value
#- List of variables
columns_stepw = step(var_dependent = 'Calories', 
                     var_independent = X.columns.to_list(), 
                     dataset = df,
                     method = 'both' ,
                     metric = 'pvalue')
columns_stepw

# %%
#- Running the model

#- Get the variable list
X_stepw_p = df[columns_stepw['var'].to_list()] 
#- Add the intercept
X_stepw_p = sm.add_constant(X_stepw_p)
#- Run the model
stepw_p = sm.OLS(y, X_stepw_p).fit()
#- Summary
print(stepw_p.summary()) 
#- Generate predicted
pred_stepw_p = stepw_p.predict(X_stepw_p)

# %%
#- Comparing the predited numbers for each different model
pred_comparative = pd.DataFrame({
'pred_model_full': pred_full,
'pred_model_step_aic': pred_stepw,
# 'pred_model_step_pvalue':pred_stepw_p,
'pred_model_backw_aic': pred_backw,
'pred_model_forw_aic': pred_forw
})
pred_comparative.head()

# %%
