# %%
# LIBRARY
import pandas as pd
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# %%
# Project Directory
project_dir = os.path.join(os.path.expanduser("~"), 
                           "OneDrive", 
                           "Project_Code", 
                           "ASN-DSA-T5", 
                           "21_CP")

# Importing Data
df = pd.read_csv(os.path.join(project_dir, "DadosAulas", "rte_cereal.csv"))
df.head()

# %%
# Column Names
col_names = {
    'Col1': 'Respondente',
    'Col2': 'Marca',
    'Col3': 'Satisfaz',
    'Col4': 'Natural',
    'Col5': 'Fibra',
    'Col6': 'Doce',
    'Col7': 'Fácil',
    'Col8': 'Sal',
    'Col9': 'Gratificante',
    'Col10': 'Energia',
    'Col11': 'Divertido',
    'Col12': 'Crianças',
    'Col13': 'Encharcado',
    'Col14': 'Econômico',
    'Col15': 'Saúde',
    'Col16': 'Família',
    'Col17': 'Calorias',
    'Col18': 'Simples',
    'Col19': 'Crocante',
    'Col20': 'Regular',
    'Col21': 'Açúcar',
    'Col22': 'Fruta',
    'Col23': 'Processo',
    'Col24': 'Qualidade',
    'Col25': 'Prazer',
    'Col26': 'Chato',
    'Col27': 'Nutritivo',
}

df0 = df.rename(columns=col_names)
df0.head()

# %%
# Q1. Rode uma análise descritiva dos dados, em relação as correlações das variáveis e avalie as correlações. Resuma estas correlações

# Descriptive Analysis - Simple Statistics
df0.describe().T


# %%
# Check combination of Respondente and Marca. Ideally we do not want duplicated entries
df0.groupby(by=['Respondente', 'Marca']).size().sort_values(ascending=False)


# %%
# Correlation Matrix

# Select variables for correlation analysis
variables = df0.columns[2:] 

CorrMatrix = df0[variables].corr()
display(CorrMatrix.round(4))


# %%
# Heatmap Visualization of Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    CorrMatrix, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    square=True, 
    center=0,
    annot_kws={"size": 8},
    cbar_kws={"shrink": .8}
)
plt.title('Matriz de Correlação das Variáveis do Cereal', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# %%
# Getting pairs of highly correlated variables
threshold = 0.4
high_corr = CorrMatrix.abs().unstack().sort_values(ascending=False)
high_corr = high_corr[high_corr < 1]  # Exclude self-correlation
high_corr = high_corr[high_corr >= threshold]
high_corr = high_corr.drop_duplicates()
high_corr.round(4)


# %%
# Q3. Obtenha os autovalores e autovetores da matriz de covariâncias
# i. Obtenha os autovalores e autovetores da matriz de correlações
# ii. Qual é a origem da diferença entre estes valores

# Covariance Matrix
CovMatrix = df0[variables].cov()

# %%
# Heatmap Visualization of Covariation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    CovMatrix, 
    annot=True, 
    fmt=".2f", 
    cmap='Blues', 
    square=True, 
    # center=0,
    annot_kws={"size": 8},
    cbar_kws={"shrink": .8}
)
plt.title('Matriz de Covariância das Variáveis do Cereal', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Eigenvalues and Eigenvectors of Covariance Matrix
eigenvalues_cov, eigenvectors_cov = np.linalg.eig(CovMatrix)

print("Autovalores da Matriz de Covariância:")
display(pd.DataFrame(eigenvalues_cov).round(4))

# %%
print("\nAutovetores da Matriz de Covariância:")
display(pd.DataFrame(eigenvectors_cov).round(4))

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(
    eigenvectors_cov, 
    annot=True, 
    fmt=".2f", 
    cmap='Purples', 
    square=True, 
    center=0,
    annot_kws={"size": 8},
    cbar_kws={"shrink": .8}
)
plt.title('Matriz de AutoVetores da Covariância', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Eigenvalues and Eigenvectors of Correlation Matrix
eigenvalues_corr, eigenvectors_corr = np.linalg.eig(CorrMatrix)

print("Autovalores da Matriz de Correlação:")
display(pd.DataFrame(eigenvalues_corr).round(4))

# %%
print("\nAutovetores da Matriz de Correlação:")
display(pd.DataFrame(eigenvectors_corr).round(4))

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(
    eigenvectors_corr, 
    annot=True, 
    fmt=".2f", 
    cmap='Purples', 
    square=True, 
    center=0,
    annot_kws={"size": 8},
    cbar_kws={"shrink": .8}
)
plt.title('Matriz de AutoVetores da Correlação', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Q4. Use as variáveis ‘Col3’ e ‘Col4’ e obtenha a matriz de covariâncias só destas 2.
df_subset = df0[['Satisfaz', 'Natural']]
CovMatrix_subset = df_subset.cov()
print("Matriz de Covariância:")
print(CovMatrix_subset.round(4))

# %%
# i. Obtenha autovalores e autovetores desta matriz simplificada, manualmente, usando (apenas treinamento de cálculo matricial)
eigenvalues_subset, eigenvectors_subset = np.linalg.eig(CovMatrix_subset)
print("\nAutovalores da Matriz de Covariância (Subset):")
print(pd.DataFrame(eigenvalues_subset).round(4))

print("\nAutovetores da Matriz de Covariância (Subset):")
print(pd.DataFrame(eigenvectors_subset).round(4))

# %%
# Q5. Monte um gráfico ordenando os autovalores.
# i. Qual é o nome deste gráfico, nas disciplinas que discutimos?
# ii. Como ele é usado?

# Scree Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues_cov) + 1),
         sorted(eigenvalues_cov, reverse=True),
         marker='o', linestyle='-')
plt.title('Gráfico dos Autovalores da Matriz de Covariância', fontsize=16)
plt.xlabel('Índice do Autovalor', fontsize=14)
plt.ylabel('Autovalor', fontsize=14)
plt.grid()
plt.xticks(range(1, len(eigenvalues_cov) + 1))
plt.tight_layout()
plt.show()

# %%
# 6. Rode uma análise de Componentes Principais com estes dados.
# i. Quantos componentes você mantem? Por quê?
# ii. Como ficam compostos os componentes selecionados?
eigenvalues_sorted = pd.DataFrame(eigenvalues_cov).sort_values(by=0, ascending=False)
eigenvalues_sorted['variance_explained'] = eigenvalues_sorted[0] / eigenvalues_sorted[0].sum()
eigenvalues_sorted['cumulative_variance'] = eigenvalues_sorted['variance_explained'].cumsum()
display(eigenvalues_sorted.round(4))

# %%
# Selected Components
n_ = 5  # Number of components to retain based on explained variance
selected_components = eigenvectors_cov[:, :n_] 
selected_components_df = pd.DataFrame(selected_components,
                                      columns=[f'PC{i+1}' for i in range(n_)],
                                      index=variables)
display(selected_components_df.round(4))

# %%
# 7. Rode uma análise Fatorial com estes dados, sem rotação
# i. O que deixamos de fazer ao não rotacionar?

# Standardizing the data
X = StandardScaler().fit_transform(df0[variables])

# Fatorial Analysis without rotation
FA_noRotation = FactorAnalysis(n_components = n_, rotation=None)
X_FA_noRotation = FA_noRotation.fit_transform(X)

# Loadings
loadings_noRotation = FA_noRotation.components_.T
loadings_noRotation_df = pd.DataFrame(loadings_noRotation,
                                       columns=[f'Factor{i+1}' for i in range(n_)],
                                        index=variables)
display(loadings_noRotation_df.round(4))

# %%
# 8. Rode uma análise fatorial com rotação varimax.
# i. O que mudou?
# ii. Como ficaram as comunalidades?
# iii. O que representam as comunalidades?

# Fatorial Analysis with Varimax rotation
FA_varimax = FactorAnalysis(n_components = n_, rotation='varimax')
X_FA_varimax = FA_varimax.fit_transform(X)

# Loadings
loadings_varimax = FA_varimax.components_.T
loadings_varimax_df = pd.DataFrame(loadings_varimax,
                                     columns=[f'Factor{i+1}' for i in range(n_)],
                                     index=variables)
display(loadings_varimax_df.round(4))

# %%
# Communalities
# No Rotation
communalities_NR = np.sum(loadings_noRotation**2, axis=1)
communalities_NR_df = pd.DataFrame(communalities_NR, 
                                 index=variables, 
                                 columns=['Communalities']) 
display(communalities_NR_df.round(4).sort_values(by='Communalities', ascending=False))

# %%
# Varimax Rotation
communalities = np.sum(loadings_varimax**2, axis=1)
communalities_df = pd.DataFrame(communalities, 
                                 index=variables, 
                                 columns=['Communalities']) 
display(communalities_df.round(4).sort_values(by='Communalities', ascending=False))

# %%
# Comunalities Chart (sorted)
communalities_df = communalities_df.sort_values(by='Communalities', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=communalities_df.index, y='Communalities', data=communalities_df)
plt.title('Comunalidades das Variáveis', fontsize=16)
plt.xlabel('Variáveis', fontsize=14)
plt.ylabel('Comunalidades', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Histogram of Communalities
plt.figure(figsize=(8, 6))
sns.histplot(communalities_df['Communalities'], bins=10, kde=True)
plt.title('Histograma das Comunalidades', fontsize=16)
plt.xlabel('Comunalidades', fontsize=14)
plt.ylabel('Frequência', fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Q9. Faça uma descrição, se for possível, dos fatores obtidos
# Loadings Interpretation
print("Loadings com Rotação Varimax:")
display(loadings_varimax_df.round(4))

# %%
# Chart for loading per factor
for i in range(n_):
    plt.figure(figsize=(10, 6))

    factor_data = loadings_varimax_df[f'Factor{i+1}'].sort_values(ascending=False)

    plot_df = pd.DataFrame({
        'variables': factor_data.index,
        'loadings': factor_data.values,
        'color_group': ['positive' if x >= 0 else 'negative' for x in factor_data.values]
    })

    ax = sns.barplot(x='variables', 
                    y='loadings',
                    hue='color_group',
                    data=plot_df,
                    palette={'positive': 'lightgreen', 'negative': 'coral'},
                    legend=False)
    
    plt.title(f'Loadings do Fator {i+1}', fontsize=16)
    plt.xlabel('Variáveis', fontsize=14)
    plt.ylabel('Loadings', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=0.5, color='blue', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
