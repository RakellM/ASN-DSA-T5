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

data = pd.read_excel(new_dir + "data/homework_02/Hospitais.xlsx")
data

alpha = 0.01

# %%
# Calculate summary statistics for the entire dataset
summary_stats = data.describe(include='all')
summary_stats

# %%
# Calculate summary statistics for each hospital
summary_stats_by_hospital = data.groupby('Hospital').describe(include='all')
summary_stats_by_hospital

# %%
# Chart to  visualize if there is a relationship between variables
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='Hospital', y='Tempo_Atendimento')
plt.title("Box Plot of Tempo Atendimento by Hospital")
plt.xlabel("Hospital")
plt.ylabel("Tempo Atendimento (min)")
plt.show()

# %%
# Normality test
# Test normality for each group
normality_results = []
for g in data['Hospital'].unique():
    sample = data[data['Hospital'] == g]['Tempo_Atendimento']
    stat, p = stats.shapiro(sample)
    normality_results.append({
        'Hospital': g,
        'statistic': stat,
        'p-value': p,
        'normal': p > 0.05  # Alpha = 0.05
    })

# Convert to DataFrame for readability
results_df = pd.DataFrame(normality_results)
print(results_df)

# %%
# Levene's Test for Homogeneity of Variances
def test_homogeneity_of_variance(data, group_col, value_col):
    groups = [group[value_col].values for name, group in data.groupby(group_col)]
    stat, p_value = stats.levene(*groups)
    alpha = 0.05
    print(f"Levene's Test Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    if p_value > alpha:
        return print("Variances are equal (fail to reject H₀)")
    else:
        return print("Variances are NOT equal (reject H₀)")

test_homogeneity_of_variance(data, 'Hospital', 'Tempo_Atendimento')
# %%
# T test for 2 independent samples
def t_test_independent_samples(data, group_col, value_col, alpha=0.05):
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError("T-test requires exactly two groups.")
    
    group1 = data[data[group_col] == groups[0]][value_col]
    group2 = data[data[group_col] == groups[1]][value_col]
    
    stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
    print(f"T-test Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    
    if p_value < alpha:
        return print("Reject H₀: Significant difference between groups")
    else:
        return print("Fail to reject H₀: No significant difference between groups")

t_test_independent_samples(data, 'Hospital', 'Tempo_Atendimento', alpha)

