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

data = pd.read_excel(new_dir + "data/homework_02/Banco.xlsx")
data

# %%
freq_A = data['A'].value_counts().sort_index()
freq_B = data['B'].value_counts().sort_index()
freq_C = data['C'].value_counts().sort_index()

frequency_table = pd.DataFrame({
    'Score': freq_A.index,
    'A': freq_A.values,
    'B': freq_B.values,
    'C': freq_C.values
})
frequency_table

# %%
frequency_table.sum(axis=0)	

# %%
frequency_table.sum(axis=1)	

# %%
# Extract the scores for each bank (A, B, C)
A_scores = data['A']
B_scores = data['B']
C_scores = data['C']

# Run the Friedman test
stat, p_value = stats.friedmanchisquare(A_scores, B_scores, C_scores)

print(f"Friedman Test Statistic: {stat:.3f}")
print(f"P-value: {p_value:.4f}")

# Check if the p-value is less than the significance level
if p_value < alpha:
    print("Reject the null hypothesis: There are significant differences between the banks.")
else:
    print("Fail to reject the null hypothesis: No significant differences between the banks.")

# %%
