# %%
import numpy as np
import pandas as pd 
from scipy import stats


# %%
n = 20
alpha = 0.05

Brand_A = 8
Brand_B = 12

# %%
# Binomial test
# test if there are no diference in the consumer preference from Brand A to Brand B

test = stats.binomtest(Brand_A, n, p=0.5 / n, alternative='two-sided')
p_value = test.pvalue
print(f"Binomial Test p-value: {p_value:.4f}")
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")

# %%




