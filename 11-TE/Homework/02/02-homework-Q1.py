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

data = pd.read_excel(new_dir + "data/homework_02/notas_precos.xlsx")
data.head()

# %%

## Cleaning Data
df = data[['nota', 'faixa_preco']]
faixa_mapping = {'g1': '0 a 5 reais',
                'g2': '5 a 10 reais',
                'g3': '10 a 15 reais',
                'g4': '15 a 20 reais',
                'g5': '20 a 25 reais',
                'g6': '25 a 30 reais'}
df['faixa_descricao'] = df['faixa_preco'].map(faixa_mapping)
df

# %%
df['faixa_preco'].unique()

# %%
df_score = df[['nota']].agg({'nota': ['mean', 'std', 'count', 'min', 'max']})
df_score

# %%
df_score1 = (df.groupby('nota')
             .agg('count')
             .sort_values(by='nota', ascending=False))
df_score1

# %%
# Summarize the data by faixa_preco
df_grouped = (df.groupby(['faixa_descricao', 'faixa_preco'])
              .agg({'nota': ['mean', 'std', 'count', 'min', 'max']})
              .sort_values(by='faixa_preco', ascending=True))
df_grouped.columns = ['mean', 'std', 'count', 'min', 'max']
df_grouped

# %%
# Chart to  visualize if there is a relationship between faixa_preco and nota
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='faixa_preco', y='nota')
plt.title("Box Plot of Score by Price Range")
plt.xlabel("Faixa Preço (Price Range)")
plt.ylabel("Nota (Score)")
plt.show()

# %%
# Test for normality
def test_normality(data):
    stat, p_value = stats.shapiro(data)
    alpha = 0.05
    if p_value > alpha:
        return True  # Data is normally distributed
    else:
        return False  # Data is not normally distributed

# %%
#test_normality(df['nota'])

# Test normality for each group
normality_results = []
for g in df['faixa_preco'].unique():
    sample = df[df['faixa_preco'] == g]['nota']
    stat, p = stats.shapiro(sample)
    normality_results.append({
        'faixa_preco': g,
        'statistic': stat,
        'p-value': p,
        'normal': p > 0.05  # Alpha = 0.05
    })

# Convert to DataFrame for readability
results_df = pd.DataFrame(normality_results)
print(results_df)

# %%
# Q-Q Plot for each group to visually assess normality
for group in df['faixa_preco'].unique():
    sample = df[df['faixa_preco'] == group]['nota']
    sm.qqplot(sample, line='s')
    plt.title(f"Q-Q Plot for Group {group}")
    plt.show()

# %%
# Distribution of nota by faixa_preco
sns.displot(data=df, x='nota', hue='faixa_preco', kind='kde', fill=True)
plt.show()

# %%
# Levene's Test for Homogeneity of Variances
def test_homogeneity_of_variance(data, group_col, value_col):
    groups = [group[value_col].values for name, group in data.groupby(group_col)]
    stat, p_value = stats.levene(*groups)
    alpha = 0.05
    if p_value > alpha:
        return print("Variances are equal (fail to reject H₀)")
    else:
        return print("Variances are NOT equal (reject H₀)")

test_homogeneity_of_variance(df, 'faixa_preco', 'nota')

# %%
# ANOVA Test
def anova_test(data, group_col, value_col):
    groups = [group[value_col].values for name, group in data.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)
    alpha = 0.05
    print(f"F-statistic: {f_stat}, p-value: {p_value}")
    if p_value > alpha:
        return print("Fail to reject H₀: No significant difference between groups")
    else:
        return print("Reject H₀: Significant difference between groups")

anova_test(df, 'faixa_preco', 'nota')

# %%
stats.f_oneway(
    df[df['faixa_preco'] == 'g1']['nota'],
    df[df['faixa_preco'] == 'g2']['nota'],
    df[df['faixa_preco'] == 'g3']['nota'],
    df[df['faixa_preco'] == 'g4']['nota'],
    df[df['faixa_preco'] == 'g5']['nota'],
    df[df['faixa_preco'] == 'g6']['nota']
)

# %%
# SQT

df1 = df.copy()

# add y_barra column with the mean of 'nota'
df1['y_barra'] = df_score['nota'].iloc[0]

# Reset index to make 'faixa_preco' a column (if needed)
df_grouped_reset = df_grouped.reset_index()

# Create a mapping dictionary: {faixa_preco: mean}
mean_mapping = dict(zip(df_grouped_reset['faixa_preco'], df_grouped_reset['mean']))

# Add y_chapeu column
df1['y_chapeu'] = df1['faixa_preco'].map(mean_mapping)

df1['y-y_barra'] = df1['nota'] - df1['y_barra']
df1['y-y_chapeu'] = df1['nota'] - df1['y_chapeu']
df1['y-y_barra_squared'] = df1['y-y_barra'] ** 2
df1['y-y_chapeu_squared'] = df1['y-y_chapeu'] ** 2

df1['y_barra-y_chapeu'] = df1['y_barra'] - df1['y_chapeu']
df1['y_barra-y_chapeu_squared'] = df1['y_barra-y_chapeu'] ** 2

# %%
df1

# %%
# SQT
SQT = df1['y-y_barra_squared'].sum()
SQT

# %%
# SQM
SQM = df1['y_barra-y_chapeu_squared'].sum()
SQM

# %%
# SQR or SQE
SQE = df1['y-y_chapeu_squared'].sum()
SQE

# %%
# proof that SQT = SQM + SQR
SQR = SQT - SQM
SQR

# %%
# R2
R2 = SQM / SQT
R2

# %%

