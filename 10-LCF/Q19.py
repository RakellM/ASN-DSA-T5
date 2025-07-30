# %%
# LIBRARY
#########################
import pandas as pd
import os
import numpy as np
#!pip install kagglehub
import kagglehub

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm 
from statsmodels.graphics.gofplots import qqplot

from scipy.stats import boxcox 
from statsmodels.stats.diagnostic import het_breuschpagan # varaince Breusch-Pagan Test

# %%
# KaggleHub
#########################
path = kagglehub.dataset_download(r'yasserh/housing-prices-dataset')

# %%
df = pd.read_csv(os.path.join(path, 'Housing.csv'))  # or whatever the actual filename is
print(df.head())

# %%
df.info()

# %%
# DATA
#########################
df1 = df[['price', 'area']]
df1.head()

# %%
# EDA
#########################
df1.describe()

# %%
# Select random sample of 50 rows
df2 = df1.sample(n=50, random_state=123)

# %%
# EDA
#########################
df2.describe()

# %%
# VISUALIZATION
#########################
## Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df2, x='area', y='price')
plt.title('Price vs Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# %%
# Create a figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, 
                               figsize=(10, 8), 
                               sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})

# Histogram - top subplot
sns.histplot(df2['price'], bins=30, kde=True, ax=ax1)
ax1.set_title('Histogram of Price')
ax1.set_xlabel('')  # Remove x-label to avoid duplication (shared x-axis)
ax1.set_ylabel('Frequency')

# Box plot - bottom subplot
sns.boxplot(data=df2, x='price', ax=ax2)
ax2.set_title('Box Plot of Price')
ax2.set_xlabel('Price')
ax2.set_ylabel('')

plt.tight_layout() # Adjust layout to prevent overlap
plt.show() # Show the plot

# %%
# Create a figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, 
                               figsize=(10, 8), 
                               sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})

# Histogram - top subplot
sns.histplot(df2['area'], bins=30, kde=True, ax=ax1)
ax1.set_title('Histogram of Area')
ax1.set_xlabel('')  # Remove x-label to avoid duplication (shared x-axis)
ax1.set_ylabel('Frequency')

# Box plot - bottom subplot
sns.boxplot(data=df2, x='area', ax=ax2)
ax2.set_title('Box Plot of Area')
ax2.set_xlabel('Area')
ax2.set_ylabel('')

plt.tight_layout() # Adjust layout to prevent overlap
plt.show() # Show the plot

# %%
# Linear Regression MODEL
#########################
# Prepare the data
X = df2[['area']]
Y = df2['price']

X = sm.add_constant(X) # Add a constant term for the intercept
model = sm.OLS(Y, X).fit()  
print(model.summary())  

# %%
# Residuals
residuals = model.resid

# Fitted values
fitted = model.fittedvalues

std_residuals = model.get_influence().resid_studentized_internal
leverage = model.get_influence().hat_matrix_diag
sqrt_abs_residuals = np.sqrt(np.abs(std_residuals))

# %%
# Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Residuals vs. Fitted
sns.scatterplot(x=fitted, y=residuals, ax=axes[0, 0])
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')

# 2. Normal Q-Q Plot
qqplot(residuals, line='s', ax=axes[0, 1], marker='o', linestyle='none')
axes[0, 1].set_title('Normal Q-Q')
axes[0, 1].set_xlabel('Theoretical Quantiles')
axes[0, 1].set_ylabel('Standardized Residuals')

# 3. Scale-Location Plot
sns.scatterplot(x=fitted, y=sqrt_abs_residuals, ax=axes[1, 0])
axes[1, 0].set_title('Scale-Location')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')

# 4. Residuals vs. Leverage
sns.scatterplot(x=leverage, y=std_residuals, ax=axes[1, 1])
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Standardized Residuals')

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# Normality test for the residuals
alpha = 0.05 # Significance level

# Perform the Shapiro-wilk test
shapiro_test = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Interpret the result
print(f"In this case, the p-value is {shapiro_test.pvalue:.4f} and with an alpha of {alpha}, we can conclude that:")

if shapiro_test.pvalue < alpha:
    print("Reject the null hypothesis that the residuals are normally distributed.")
else:
    print("Fail to reject the null hypothesis that the residuals are normally distributed.")


# %%
# Transformation
#########################
# Apply Box-Cox transformation to the 'price' variable
df2['price_boxcox'], lambda_value = boxcox(df2['price'])

# Print the lambda value
print(f"Lambda value for Box-Cox transformation: {lambda_value}")

# %%
# Linear Regression MODEL with transformed data
X = df2[['area']]
Y_transformed = df2['price_boxcox']
X = sm.add_constant(X)  # Add a constant term for the intercept 

model_transformed = sm.OLS(Y_transformed, X).fit()  
print(model_transformed.summary())  

# %%
# Residuals
residuals = model_transformed.resid

# Fitted values
fitted = model_transformed.fittedvalues

std_residuals = model_transformed.get_influence().resid_studentized_internal
leverage = model_transformed.get_influence().hat_matrix_diag
sqrt_abs_residuals = np.sqrt(np.abs(std_residuals))

# %%
# Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Residuals vs. Fitted
sns.scatterplot(x=fitted, y=residuals, ax=axes[0, 0])
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')

# 2. Normal Q-Q Plot
qqplot(residuals, line='s', ax=axes[0, 1], marker='o', linestyle='none')
axes[0, 1].set_title('Normal Q-Q')
axes[0, 1].set_xlabel('Theoretical Quantiles')
axes[0, 1].set_ylabel('Standardized Residuals')

# 3. Scale-Location Plot
sns.scatterplot(x=fitted, y=sqrt_abs_residuals, ax=axes[1, 0])
axes[1, 0].set_title('Scale-Location')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')

# 4. Residuals vs. Leverage
sns.scatterplot(x=leverage, y=std_residuals, ax=axes[1, 1])
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Standardized Residuals')

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# Perform the Shapiro-wilk test
shapiro_test = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Interpret the result
print(f"In this case, the p-value is {shapiro_test.pvalue:.4f} and with an alpha of {alpha}, we can conclude that:")

if shapiro_test.pvalue < alpha:
    print("Reject the null hypothesis that the residuals are normally distributed.")
else:
    print("Fail to reject the null hypothesis that the residuals are normally distributed.")

# %%
### Statistical Tests for Heteroscedasticity
#- Breusch-Pagan Test (Best for Linear Models)
bp_test = het_breuschpagan(model_transformed.resid, model_transformed.model.exog)
print(f"Breusch-Pagan Test:")
print(f"Lagrange Multiplier: {bp_test[0]:.3f}, p-value: {bp_test[1]:.3f}")

# Interpret the result
print(f"In this case, the p-value is {bp_test[1]:.4f} and with an alpha of {alpha}, we can conclude that:")

if bp_test[1] < alpha:
    print("Reject the null hypothesis that the residuals are distributed with equal variance\n(heteroscedasticity is present).")
else:
    print("Fail to reject the null hypothesis that the residuals are distributed with equal variance\n(homoscedasticity is present).")


# %%
