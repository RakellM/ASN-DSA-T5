# %%

# Uma startup quer saber se o novo modelo de precificação reduziu o tempo médio de negociação com clientes. Foram medidos os tempos (em minutos) antes e depois para os mesmos 8 vendedores:
#Antes: [45, 50, 48, 52, 47, 49, 51, 46]
#Depois: [39, 41, 42, 44, 40, 43, 42, 39]

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# %%

before = pd.Series([45, 50, 48, 52, 47, 49, 51, 46])
after = pd.Series([39, 41, 42, 44, 40, 43, 42, 39])

# H0: mu_before - mu_after = 0
# H1: mu_before - mu_after < 0

H1 = "less" # greater, less or different
alpha = 0.05

# %%
# Calculate descriptive statistics
before.describe()

# %%
after.describe()

# %%

# Normality test
# Use Shapiro-Wilk test to check if the data is normally distributed, as n is less than 31
sw_test_before = stats.shapiro(before)
sw_test_after = stats.shapiro(after)
print("Shapiro-Wilk test:")
print(f"BEFORE: Statistic: {sw_test_before.statistic:.2f}, p-value: {sw_test_before.pvalue:.2f}")
print(f"AFTER: Statistic: {sw_test_after.statistic:.2f}, p-value: {sw_test_after.pvalue:.2f}")

# %%

# Histogram of the data
#-- Before
plt.hist(before, bins=5, density=True, alpha=0.5, color='blue')
plt.title('Histogram of Time BEFORE (min)')
plt.show()

# %%
#-- After
plt.hist(after, bins=5, density=True, alpha=0.5, color='orange')
plt.title('Histogram of Time AFTER (min)')
plt.show()

# %%
# Combined histogram for comparison
plt.hist(before, bins=5, alpha=0.5, label='Before', color='blue', density=True)
plt.hist(after, bins=5, alpha=0.5, label='After', color='orange', density=True)
plt.title('Histogram Comparison: Before vs After')
plt.xlabel('Time (min)')
plt.ylabel('Density')
plt.legend()
plt.show()


# %%

# Box-Plot of the data
plt.figure(figsize=(6, 4))
plt.boxplot([before, after], vert=True, labels=['Before', 'After'])
plt.title('Box-Plot of Negotiation Time')
plt.ylabel('Time (min)')
plt.show()

# %%
# Levene's test for equal variances
levene_test = stats.levene(before, after)
print("Levene's test for equal variances:")
print(f"Statistic: {levene_test.statistic:.2f}, p-value: {levene_test.pvalue:.2f} \n")

if levene_test.pvalue < alpha:
    print("Reject H0: The variances are different")
else:
    print("Do not reject H0: The variances are equal")

# %%

# t-test for paired samples where H1 is different (two-tailed)
t_test = stats.ttest_rel(before, after)
print("Paired t-test:")
print(f"t-statistic: {t_test.statistic:.2f}, p-value: {t_test.pvalue:.2f} \n")
if t_test.pvalue < alpha:
    print("Reject H0: The mean time is different")
else:
    print("Do not reject H0: The mean time is equal")

# %%

# t-test for paired samples where H1 is one tailed
t_test_one_tailed = stats.ttest_rel(before, after, alternative='greater')
print("Paired t-test (one-tailed):")
print(f"t-statistic: {t_test_one_tailed.statistic:.2f}, p-value: {t_test_one_tailed.pvalue:.2f} \n")
if t_test_one_tailed.pvalue < alpha:
    print(f"Reject H0: The mean time difference is {H1} than 0")
else:
    print("Do not reject H0: The mean time is equal.")

