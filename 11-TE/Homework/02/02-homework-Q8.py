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

data = pd.read_excel(new_dir + "data/homework_02/Dieta.xlsx")
data

# %%
# Calculate summary statistics for the entire dataset
summary_stats = data.describe(include='all')
summary_stats

# %%
# Chart to  visualize if there is a relationship between variables
plt.figure(figsize=(8, 6))
sns.boxplot(data=data[['Antes', 'Depois']])
plt.title("Box Plot: Peso Antes e Depois da Dieta")
plt.ylabel("Peso (kg)")
plt.show()

# %%
# Histogram to visualize the distribution of weights before and after the diet
plt.figure(figsize=(12, 6))
sns.histplot(data['Antes'], kde=True, color='blue', label='Antes', stat='density', bins=20)
sns.histplot(data['Depois'], kde=True, color='orange', label='Depois', stat='density', bins=20)
plt.title("Histogram: Peso Antes e Depois da Dieta")
plt.xlabel("Peso (kg)")
plt.ylabel("Density")
plt.legend()
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
print("Normality Test Results - Before Diet:")
test_normality(data['Antes'])

# %%
print("\nNormality Test Results - After Diet:")
test_normality(data['Depois'])

# %%
# Wilcoxon Signed-Rank Test for Paired Samples
wilcoxon_stat, wilcoxon_p_value = stats.wilcoxon(data['Antes'], data['Depois'], alternative='greater')
print(f"Wilcoxon Signed-Rank Test Statistic: {wilcoxon_stat:.4f}, p-value: {wilcoxon_p_value:.4f}")

if wilcoxon_p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference before and after.")
else:
    print("Fail to reject the null hypothesis: No significant difference before and after.")

# %%
data1 = data.copy()
data1['Diferenca'] = data1['Depois'] - data1['Antes']
data1

# %%
# Histogram to visualize the distribution of differences
plt.figure(figsize=(8, 6))
sns.histplot(data1['Diferenca'], kde=True, color='green', stat='density', bins=20)
plt.title("Histogram: Diferença de Peso Antes e Depois da Dieta")
plt.xlabel("Diferença de Peso (kg)")
plt.ylabel("Density")
plt.axvline(0, color='red', linestyle='--', label='Zero Difference')
plt.legend()
plt.show()

# %%
print("\nNormality Test Results - Difference:")
test_normality(data1['Diferenca'])

# %%
data1a = data1[data1['Diferenca'] != 0]     # Remove zero differences for the Wilcoxon test
data1a

# %%
# Calculate summary statistics for the entire dataset
summary_stats1 = data1a.describe(include='all')
summary_stats1

# %%
# Wilcoxon Signed-Rank Test for Paired Samples
wilcoxon_stat, wilcoxon_p_value = stats.wilcoxon(data1['Antes'], data1['Depois'], alternative='greater')
print(f"Wilcoxon Signed-Rank Test Statistic: {wilcoxon_stat:.4f}, p-value: {wilcoxon_p_value:.4f}")

if wilcoxon_p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference before and after.")
else:
    print("Fail to reject the null hypothesis: No significant difference before and after.")

# %%
