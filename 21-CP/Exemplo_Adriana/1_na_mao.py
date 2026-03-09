# %%
# Library

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer  

# %%
# Project Directory
project_dir = os.path.join(os.path.expanduser("~"), 
                           "OneDrive", 
                           "Project_Code", 
                           "ASN-DSA-T5", 
                           "21_CP", 
                           "Exemplo_Adriana")

df = pd.read_excel(os.path.join(project_dir, "data", "mao.xlsx"))
df.head()

# %%
# 1. Calculate Correlation Matrix
correlation_matrix = df.corr()
print(correlation_matrix)


# %%
# visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
# 2. Calculate KMO
kmo_all, kmo_model = calculate_kmo(df)
print(f"KMO Model: {kmo_model}")

# %% 
# 3. Perform Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(df)
print(f"Bartlett's Test Chi-Square: {chi_square_value}, p-value: {p_value}")


# %%
# 4. Calculate Principal Components
fa = FactorAnalyzer(n_factors = 2, rotation = None)
fa.fit(df)
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)

eigen_values, vectors = fa.get_eigenvalues()
print("Eigenvalues:\n", eigen_values)


# %%
# biplot chart
plt.figure(figsize=(8, 6))
plt.scatter(loadings[:, 0], loadings[:, 1])
for i, txt in enumerate(df.columns):
    plt.annotate(txt, (loadings[i, 0], loadings[i, 1]))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axhline(0, color='blue', lw=2)
plt.axvline(0, color='blue', lw=2)  
plt.title('Biplot of Factor Loadings (no rotation)')
plt.grid()
plt.show()



# %%
# Factor Loadings
fa = FactorAnalyzer(n_factors = 2, rotation = "varimax")
fa.fit(df)
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)



# %%
# biplot chart
# biplot chart
plt.figure(figsize=(8, 6))
plt.scatter(loadings[:, 0], loadings[:, 1])
for i, txt in enumerate(df.columns):
    plt.annotate(txt, (loadings[i, 0], loadings[i, 1]))
plt.xlabel('RC1')
plt.ylabel('RC2')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axhline(0, color='blue', lw=2)
plt.axvline(0, color='blue', lw=2)  
plt.title('Biplot of Factor Loadings (with rotation)')
plt.grid()
plt.show()



# %%
# Autovalues
eigen_values, vectors = fa.get_eigenvalues()
print("Eigenvalues:\n", eigen_values)


# %%
# % Variance Explained
variance_explained = fa.get_factor_variance()
# print("Variance Explained:\n", variance_explained)
display(variance_explained)

# %%
# Factor Scores (weights)
factor_scores = fa.transform(df)
print("Factor Scores:\n", factor_scores)

# %%
# Loadings Plot
plt.figure(figsize=(8, 6))
plt.bar(range(len(loadings)), loadings[:, 0], label='RC1', alpha=0.7)
plt.bar(range(len(loadings)), loadings[:, 1], label='RC2', alpha=0.7, bottom=loadings[:, 0])
plt.xticks(range(len(loadings)), df.columns, rotation=45)
plt.ylabel('Loadings')
plt.title('Factor Loadings for RC1 and RC2')
plt.legend()
plt.tight_layout()
plt.show()


# %%
