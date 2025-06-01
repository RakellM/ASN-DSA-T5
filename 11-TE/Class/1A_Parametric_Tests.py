# Goal Undestand Parametric Tests in Python

# %%
## Library

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# %%
os.getcwd()
new_dir = "C:/Users/kel_m/OneDrive/Nerd_Code/ASN-DSA-T5/11-TE/" 
os.chdir(new_dir)

# %%
## 1. Normality Test
#- create a variable x with 1000 random numbers from a normal distribution with mean 0 and standard deviation 1
x = pd.Series(np.random.normal(0, 1, 1000))

# %%
#- plot an histogram of x with 30 bins and a density plot
plt.hist(x, bins=30, density=True, alpha=0.5, color='blue')

# %%
#- normality test with scipy.stats.shapiro

shapiro_test = stats.shapiro(x)
shapiro_test

# %%
#- normality test with kolmogorov-smirnov test
ks_test = stats.kstest(x, 'norm', args=(x.mean(), x.std()))
ks_test

# %%
## 2. Variance Test
variance = pd.read_excel(new_dir + "data/variancia.xlsx")

# %%
variance.describe()

# %%
v1_x = variance['medida'][variance['grupo'] == 'grupo1']
plt.hist(v1_x, bins=30, density=True, alpha=0.5, color='blue')

# %%
v2_x = variance['medida'][variance['grupo'] == 'grupo2']
plt.hist(v1_x, bins=30, density=True, alpha=0.5, color='green')

