# %%
import numpy as np
import pandas as pd 
from scipy import stats

# %%
n = 60
alpha = 0.05

Book_A = 29
Book_B = 15
Book_C = 16

# %%
# Chi-squared test
observed = np.array([Book_A, Book_B, Book_C])
expected = np.array([n/3, n/3, n/3])  # Assuming equal distribution

chi2_stat, p_value = stats.chisquare(observed, expected)
print(f"Chi-squared Test Statistic: {chi2_stat:.4f}, p-value: {p_value:.4f}")
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")

# %%
