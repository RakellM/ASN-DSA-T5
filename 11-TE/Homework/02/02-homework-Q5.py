# %%
## Library

import os
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# %%

os.getcwd()
new_dir = "C:/Users/kel_m/OneDrive/Nerd_Code/ASN-DSA-T5/11-TE/" 
os.chdir(new_dir)

# %%
alpha = 0.05

data = pd.read_excel(new_dir + "data/homework_02/Colesterol.xlsx")
data

# %%
# Calculate summary statistics for the entire dataset
summary_stats = data.describe(include='all')
summary_stats

# %%
# Chart to  visualize if there is a relationship between variables
plt.figure(figsize=(8, 6))
sns.boxplot(data=data[['Antes_tratamento', 'Depois_tratamento']])
plt.title("Box Plot: Antes e Depois do Tratamento vs Colesterol")
plt.ylabel("Colesterol LDL (mg/dL)")
plt.show()

# %%
# Normality test
def test_normality(data):
    stat, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk Test Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    alpha = 0.05
    if p_value > alpha:
        return True  # Data is normally distributed
    else:
        return False  # Data is not normally distributed

# %%
print("Normality Test Results - Before Treatment:")
test_normality(data['Antes_tratamento'])

# %%
print("\nNormality Test Results - After Treatment:")
test_normality(data['Depois_tratamento'])

# %%
# Levene's Test for Homogeneity of Variances
levene_stat, levene_p_value = stats.levene(data['Antes_tratamento'], data['Depois_tratamento'])
print(f"Levene's Test Statistic: {levene_stat:.4f}, p-value: {levene_p_value:.4f}")
alpha = 0.05
if levene_p_value > alpha:
    print("Variances are equal (fail to reject H₀)")
else:
    print("Variances are NOT equal (reject H₀)")

# %%
# T test for 2 paired samples
def t_test_paired_samples(data, before_col, after_col, alpha=0.05):
    if len(data[before_col]) != len(data[after_col]):
        raise ValueError("Before and after columns must have the same length.")
    
    stat, p_value = stats.ttest_rel(data[before_col], data[after_col], alternative="greater")
    print(f"T-test Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    
    if p_value < alpha:
        return print("Reject H₀: Significant difference between before and after")
    else:
        return print("Fail to reject H₀: No significant difference between before and after")

t_test_paired_samples(data, 'Antes_tratamento', 'Depois_tratamento', alpha)


# %%
