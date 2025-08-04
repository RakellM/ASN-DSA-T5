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
from linear_regression_functions import step, charts_var_num, plot_regression_diagnostics
# import linear_regression_functions

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
                         figsize=(2, 1), 
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
## MODEL Forward
### AIC
# Independent Variables (Matrix)
X = df.drop(["Calories", "User_ID", "Gender"] , axis=1)
# Dependent Variable (vector)
y = df["Calories"]

columns_forw = step(var_dependent = 'Calories', 
                    var_independent = X.columns.to_list(), 
                    dataset = df, 
                    method = 'forward', 
                    metric = 'aic') 
columns_forw

# %%
X_forw = df[columns_forw['var'].to_list()[0] ]

X_forw = sm.add_constant(X_forw)

forw = sm.OLS(y, X_forw).fit()

print(forw.summary()) 

pred_forw = forw.predict(X_forw)

#### Done


# %%
### BIC
columns_back = step(var_dependent = 'Calories', 
                    var_independent = X.columns.to_list(), 
                    dataset = df, 
                    method = 'forward', 
                    metric = 'bic') 
columns_back

# %%
### p-value
columns_back = step(var_dependent = 'Calories', 
                    var_independent = X.columns.to_list(), 
                    dataset = df, 
                    method = 'backward', 
                    metric = 'pvalue') 
columns_back
# %%
