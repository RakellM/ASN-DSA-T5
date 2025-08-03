# %%
# LIBRARY
#!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from scipy import stats # Shapiro-Wilk Normality Test
from scipy.stats import boxcox 
from statsmodels.stats.diagnostic import het_breuschpagan # varaince Breusch-Pagan Test

# %%
# DATASET
# https://archive.ics.uci.edu/ml/datasets/auto+mpg
#- fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
#- data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 
  
#- metadata 
print(auto_mpg.metadata) 
  
#- variable information 
print(auto_mpg.variables) 

# %%
df = auto_mpg.data.original
df.info()

# %%
# DATA PREPROCESSING
df.head()
df1 = df.copy()

# %%
## EDA
#- Check for missing values
missing_values = df1.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# %%
#- Count number of cylinders
cylinder_counts = df1['cylinders'].value_counts()  
print("Count of cylinders:")
print(cylinder_counts)

# %%
#- Count origin categories
origin_counts = df1['origin'].value_counts()
print("Count of origin categories:")
print(origin_counts)

# %%
#- Count model years
model_year_counts = df1['model_year'].value_counts()
print("Count of model years:")
print(model_year_counts)

# %%
#- Summary statistics
summary_stats = df1.describe()
print("Summary statistics:")
print(summary_stats)

# %%
#- one variable at a time from the summary_stats
summary_stats['mpg']

# %%
## DATA VISUALIZATION
df_plot = df1.copy()
variable = 'displacement'  # Variable to plot

#- Histogram & Boxplot
fig, (ax1, ax2) = plt.subplots(2, 1, 
                               figsize=(6, 4), 
                               sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})

# Histogram - top subplot
sns.histplot(df_plot[variable], bins=30, kde=True, ax=ax1)
ax1.set_title(f'Histogram of {variable}')
ax1.set_xlabel('')  # Remove x-label to avoid duplication (shared x-axis)
ax1.set_ylabel('Frequency')

# Box plot - bottom subplot
sns.boxplot(data=df_plot, x=variable, ax=ax2)
ax2.set_title(f'Box Plot of {variable}')
ax2.set_xlabel(variable)
ax2.set_ylabel('')

plt.tight_layout() # Adjust layout to prevent overlap
plt.show() # Show the plot

# %%
## DATA PREPROCESSING
df2 = df1.copy()

#- Convert 'origin' to categorical
df2['origin'] = df2['origin'].astype('category')

#- Create origin_desc column
df2['origin_desc'] = df2['origin'].cat.rename_categories({1: '1_USA',
                                                          2: '2_Europe',
                                                          3: '3_Japan'})

df2['origin2'] = df2['origin'].astype('int64').replace({1: 1, 2: 2, 3: 2})

df2['origin_US'] = df2['origin2'].astype('category').cat.rename_categories({1: '1_USA',
                                                         2: '2_NonUS',
                                                         3: '2_NonUS'})

df2['cylinders2'] = df2['cylinders'].astype('int64').replace({3: 4, 4: 4, 5: 4, 6: 6, 8: 8})

#- Create Age of the car comparing to 2025
df2['car_age'] = 2025 - (1900 + df2['model_year'])

# %%
### Feature Engineering
#- Create new uncorrelated features
#- Power to Weight Ratio
df2['power_to_weight'] = df2['horsepower'] / df2['weight'] 

#- Engine Stress
df2['engine_stress'] = df2['displacement'] / df2['horsepower'] 

#- Cylinder Optimization
df2['cylinder_opt'] = df2['horsepower'] / ( df2['cylinders'] * df2['displacement'] )

#- Dynamic Response
df2['dynamic_response'] = df2['acceleration'] / df2['weight']  

#- Structural Factor
df2['structural_factor'] = np.log( df2['weight'] / df2['displacement'] )

#- Vehicle class
df2['vehicle_class'] = pd.cut(df2['weight'], 
                           bins=[0, 2500, 3500, float('inf')],
                           labels=['1_light', '2_medium', '3_heavy'])

# %%
#### Box-Cox transformation
#- Apply Box-Cox transformation to the 'mpg' variable
df2['mpg_boxcox'], lambda_value = boxcox(df2['mpg'])

#- Print the lambda value
print(f"Lambda value for Box-Cox transformation: {lambda_value}")

# %%
#- Count of observations per level
vehicle_class_counts = df2['vehicle_class'].value_counts()
print("Count of vehicle class:")
print(vehicle_class_counts)

# %%
#### Dummys
#- Convert categorical to dummy variables
df3 = pd.get_dummies(df2, 
                     columns=['cylinders2', 'origin_desc', 'vehicle_class', 'origin_US'], 
                     drop_first=True)

df3.info()

# %%
df4 = df3.copy()

#- Remove missings from horsepower
df4 = df4.dropna(subset=['horsepower'])

#- Remove non-predictive columns
#- play with which variables are in the model
df4 = df4.drop(['car_name', 
                'origin', 
                'cylinders', 
                'model_year',
                'displacement',
                'horsepower',
                'weight',
                'acceleration',
                # 'car_age',
                # 'power_to_weight',
                'engine_stress',
                # 'cylinder_opt',
                'dynamic_response',
                'structural_factor' ,
                'cylinders2_6',
                'cylinders2_8',
                # 'origin_desc_2_Europe',
                # 'origin_desc_2_Japan',
                # 'vehicle_class_2_medium',
                # 'vehicle_class_3_heavy' ,
                'origin2' ,
                'origin_US_2_NonUS' ,
                # 'mpg_boxcox' ,
                'mpg'
                ], axis=1)

df4.info()

# %%
### Correlation Matriz
#- Plot correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df4.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# %%
#- Identify high correlations (absolute value > 0.7)
high_corr = df4.corr().abs().stack().reset_index()
high_corr = high_corr[high_corr[0] > 0.7]
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
print("\nHighly Correlated Features:")
print(high_corr)

# %%
### VIF
#- Calculate VIF for each feature
var_Y = 'mpg_boxcox'
X_vif = df4.drop(var_Y, axis=1).select_dtypes(include=['int64','float64'])
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values.astype('float'), i) 
                  for i in range(X_vif.shape[1])]

print("Variance Inflation Factors:")
print(vif_data.sort_values("VIF", ascending=False))

#- Remove features with VIF > 5
high_vif = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
df_filtered = df4.drop(high_vif, axis=1)
print(f"\nRemoved features due to high VIF: {high_vif}")


# %%
## MODEL
### Model 1
X = df3.drop('mpg', axis=1).astype('float64')
X = sm.add_constant(X) 
y = df3['mpg']

model = sm.OLS(y, X).fit()
print(model.summary())

# %%
### Model 2
#- remove power-to-weight
X = X.drop('power_to_weight', axis=1)
# X = X.drop('cylinder_opt', axis=1)
# X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()
print(model.summary())

# %%
### Model 3
X = df4.drop(['mpg_boxcox'], axis=1).astype('float64')
X = sm.add_constant(X) 
y = df4['mpg_boxcox']

model = sm.OLS(y, X).fit()
print(model.summary())

# %%
### Model 4
X = df4.drop(['mpg_boxcox', 'cylinder_opt'], axis=1).astype('float64')
X = sm.add_constant(X) 
y = df4['mpg_boxcox']

model = sm.OLS(y, X).fit()
print(model.summary())

# %%
## Residual Analysis
#- Residuals
residuals = model.resid

#- Fitted values
fitted = model.fittedvalues

std_residuals = model.get_influence().resid_studentized_internal
leverage = model.get_influence().hat_matrix_diag
sqrt_abs_residuals = np.sqrt(np.abs(std_residuals))

# %%
### Residual Analysis Plots
#- Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

#- 1. Residuals vs. Fitted
sns.scatterplot(x=fitted, y=residuals, ax=axes[0, 0])
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')

#- 2. Normal Q-Q Plot
qqplot(residuals, line='s', ax=axes[0, 1], marker='o', linestyle='none')
axes[0, 1].set_title('Normal Q-Q')
axes[0, 1].set_xlabel('Theoretical Quantiles')
axes[0, 1].set_ylabel('Standardized Residuals')

#- 3. Scale-Location Plot
sns.scatterplot(x=fitted, y=sqrt_abs_residuals, ax=axes[1, 0])
axes[1, 0].set_title('Scale-Location')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')

#- 4. Residuals vs. Leverage
sns.scatterplot(x=leverage, y=std_residuals, ax=axes[1, 1])
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Standardized Residuals')

#- Adjust layout
plt.tight_layout()
plt.show()

# %%
### Normality Test
#- Perform the Shapiro-wilk test
shapiro_test = stats.shapiro(residuals)
alpha = 0.05
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue} \n")

# Interpret the result
print(f"In this case, the p-value is {shapiro_test.pvalue:.4f} and with an alpha of {alpha}, we can conclude that:")

if shapiro_test.pvalue < alpha:
    print("Reject the null hypothesis that the residuals are normally distributed.")
else:
    print("Fail to reject the null hypothesis that the residuals are normally distributed.")

# %%
### Statistical Tests for Heteroscedasticity
#- Breusch-Pagan Test (Best for Linear Models)
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan Test:")
print(f"Lagrange Multiplier: {bp_test[0]:.3f}, p-value: {bp_test[1]:.3f}")

# Interpret the result
print(f"In this case, the p-value is {bp_test[1]:.4f} and with an alpha of {alpha}, we can conclude that:")

if bp_test[1] < alpha:
    print("Reject the null hypothesis that the residuals are distributed with equal variance\n(heteroscedasticity is present).")
else:
    print("Fail to reject the null hypothesis that the residuals are distributed with equal variance\n(homoscedasticity is present).")


# %%

