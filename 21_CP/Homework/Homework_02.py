# %%
# LIBRARY
import pandas as pd
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# Project Directory
project_dir = os.path.join(os.path.expanduser("~"), 
                           "OneDrive", 
                           "Project_Code", 
                           "ASN-DSA-T5", 
                           "21_CP")

# Importing Data
df = pd.read_csv(os.path.join(project_dir, "DadosExercicios", "bodyfat-reduced.csv"))
df.head()

# %%
# Q10. Use bodyfat-reduced.csv e faça uma análise de Componentes principais.

# Descriptive statistics
df.describe().T.round(4)

# %%
# Standardizing the data
features = df.columns.tolist()
features.remove('BodyFat')
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)

# %%
# PCA Analysis
pca = PCA()
principalComponents = pca.fit_transform(x)

variance_pca = pca.explained_variance_ratio_

plt.figure(figsize=(8,5))
plt.plot(range(1, len(variance_pca)+1), np.cumsum(variance_pca), marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Principal Component Numbers')
plt.ylabel('Cumulative Variance')
plt.grid(True)
plt.show()
     
# %%
# PCA with 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
principalDf.head()

# %%
# Loadings
loadings = pca.components_.T

var_names = list(df.columns)

# %%
# Visualizing the PCA result
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.7)
plt.axhline(0, linewidth=1, alpha=0.4, color='red')
plt.axvline(0, linewidth=1, alpha=0.4, color='red')
plt.title('PCA Result with 2 Components (not scaled)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# %%
# Q11. Faça um gráfico biplot da análise de componentes principais acima.
# Remove the first variable (target) from var_names
var_names_features = var_names[1:]  # This should now have 6 variables

# Scaling the loadings for better visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=principalDf, alpha=0.7, s=60)
plt.axhline(0, linewidth=1, alpha=0.4, color='red')
plt.axvline(0, linewidth=1, alpha=0.4, color='red') 

scale_X = principalComponents[:, 0].max() - principalComponents[:, 0].min()
scale_Y = principalComponents[:, 1].max() - principalComponents[:, 1].min()

arrow_scale = 0.4


for i, var in enumerate(var_names_features):
    dx = loadings[i, 0] * scale_X * arrow_scale
    dy = loadings[i, 1] * scale_Y * arrow_scale

    plt.arrow(0, 0, dx, dy, 
              color='green', 
              alpha=0.5, 
              head_width=0.1,
              length_includes_head=True)
    
    plt.text(dx * 1.05, dy * 1.05, 
             var, 
             color='black', 
             ha='center', 
             va='center')

pc1_pct = variance_pca[0] * 100
pc2_pct = variance_pca[1] * 100

plt.title('PCA Biplot')
plt.xlabel(f"Principal Component 1 ({pc1_pct:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pc2_pct:.1f}%)")
plt.grid(True, alpha=0.3)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.show()


# %%
