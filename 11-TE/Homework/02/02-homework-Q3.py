# %%

from scipy import stats
import numpy as np

mu_0 = 60
mu = 65
n = 36
s = 3.5
alpha = 0.05

# Alternative way: using scipy.stats.ttest_1samp (for t-test with unknown population std)

np.random.seed(0)
sample = np.random.normal(loc=mu, scale=s, size=n)
t_stat, p_value = stats.ttest_1samp(sample, popmean=mu_0)
print(f"\nUsing scipy.stats.ttest_1samp:")
print(f"t-statistic: {t_stat:.4f}")
# print(f"p-value: {p_value/2:.4f} (one-tailed)")  # divide by 2 for one-tailed test
print(f"p-value: {p_value:.4f} (two-tailed)")  

if p_value < alpha:
    print("Reject the null hypothesis (H₀)")
else:
    print("Fail to reject the null hypothesis (H₀)")


# %%