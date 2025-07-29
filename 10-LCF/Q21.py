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
variable = 'mpg'  # Variable to plot

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
#- Count of observations per level
vehicle_class_counts = df2['vehicle_class'].value_counts()
print("Count of vehicle class:")
print(vehicle_class_counts)

# %%
#### Dummys
#- Convert categorical to dummy variables
df2 = pd.get_dummies(df2, columns=['cylinders2', 'origin_desc', 'vehicle_class'], drop_first=True)

df2.info()

# %%
df3 = df2.copy()

#- Remove missings from horsepower
df3 = df3.dropna(subset=['horsepower'])

#- Remove non-predictive columns
df3 = df3.drop(['car_name', 
                'origin', 
                'cylinders', 
                'model_year',
                #'displacement',
                #'horsepower',
                #'weight',
                #'acceleration',
                #'structural_factor' ,
                #'engine_stress'
                ], axis=1)

# %%
### Correlation Matriz
#- Plot correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# %%
#- Identify high correlations (absolute value > 0.7)
high_corr = df3.corr().abs().stack().reset_index()
high_corr = high_corr[high_corr[0] > 0.7]
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
print("\nHighly Correlated Features:")
print(high_corr)

# %%
### VIF
#- Calculate VIF for each feature
X_vif = df3.drop('mpg', axis=1).select_dtypes(include=['int64','float64'])
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values.astype('float'), i) 
                  for i in range(X_vif.shape[1])]

print("Variance Inflation Factors:")
print(vif_data.sort_values("VIF", ascending=False))

#- Remove features with VIF > 5
high_vif = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
df_filtered = df3.drop(high_vif, axis=1)
print(f"\nRemoved features due to high VIF: {high_vif}")




# %%
