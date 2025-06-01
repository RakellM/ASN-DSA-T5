# %%

# Uma empresa quer saber se a nota média da satisfação dos clientes após implementar um novo atendimento via chatbot é diferente de 8 (padrão antigo). A nota é de 0 a 10. Foram coletadas as notas de 12 clientes:
# [7.5, 8.1, 8.4, 7.8, 8.0, 8.3, 8.2, 7.9, 8.5, 8.6, 7.7, 8.1]

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# %%

score = pd.Series([7.5, 8.1, 8.4, 7.8, 8.0, 8.3, 8.2, 7.9, 8.5, 8.6, 7.7, 8.1])
score

# %%
# Calculate descriptive statistics
score.describe()

# %%
mode = score.mode()[0]
mode

# %%

# Normality test
# Use Shapiro-Wilk test to check if the data is normally distributed, as n is less than 31
shapiro_test = stats.shapiro(score)
shapiro_test


# %%

# Histogram of the data
#-- Bin = 30
plt.hist(score, bins=30, density=True, alpha=0.5, color='blue')

# %%
#-- Bin = 10
plt.hist(score, bins=10, density=True, alpha=0.5, color='blue')

# %%

# Box-Plot of the data
plt.boxplot(score, vert=False)
plt.title('Box-Plot of the Score')


# %%
# Back to the problem
# Hipothesis test
# H0: mu = 8
# H1: mu != 8

mu = 8
alpha = 0.05
H1 = "different" # greater, less or different

## Compare statistics with t-distribution
#-- t critical value
t_critical = stats.t.ppf(1 - alpha/2, len(score) - 1)
t_critical

#-- t calculated value
t_calculate = (score.mean() - mu) / (score.std() / np.sqrt(len(score)))
t_calculate

print(f"t_calculate: {t_calculate}")
print(f"t_critical: {t_critical}")

# %%
## Confidence Interval for mu with 95% of confidence

#-- t critical value
t_critical = stats.t.ppf(1 - alpha/2, len(score) - 1)
t_critical

#-- standard error
standard_error = score.std() / np.sqrt(len(score))
standard_error

#-- confidence interval
confidence_interval = (mu - t_critical * standard_error, mu + t_critical * standard_error)
confidence_interval

# %%
## p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_test.statistic), len(score) - 1))
p_value	

#-- p-value > alpha, do not reject H0

# %%

## Confidence Interval for x_bar with 95% of confidence

#-- t critical value    
t_critical = stats.t.ppf(1 - alpha/2, len(score) - 1)
t_critical

#-- standard error
standard_error = score.std() / np.sqrt(len(score))
standard_error

#-- confidence interval
confidence_interval = (score.mean() - t_critical * standard_error, 
                       score.mean() + t_critical * standard_error)
confidence_interval

# %%
# Visualization 1: Histogram with mean and mu
plt.figure(figsize=(6, 4))
plt.hist(score, bins=10, alpha=0.5, color='skyblue', edgecolor='black', density=True)
plt.axvline(score.mean(), color='green', linestyle='--', label='Sample Mean')
plt.axvline(mu, color='red', linestyle='-', label='Mu (8)')
plt.title('Histogram with Mean and Mu')
plt.legend()
plt.show()

# %%

# Visualization 2: Boxplot with mu
plt.figure(figsize=(6, 2))
plt.boxplot(score, vert=False)
plt.axvline(mu, color='red', linestyle='--', label='Mu (8)')
plt.title('Boxplot with Mu')
plt.legend()
plt.show()

# %%

# Visualization 3: t-distribution with critical regions and t_calculate
x = np.linspace(-4, 4, 200)
y = stats.t.pdf(x, df=len(score)-1)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='t-distribution')
plt.fill_between(x, y, where=(x <= -t_critical) | (x >= t_critical), color='red', alpha=0.3, label='Rejection Region')
plt.axvline(t_calculate, color='orange', linestyle='--', label='t_calculate')
plt.axvline(-t_critical, color='black', linestyle=':', label='-t_critical')
plt.axvline(t_critical, color='black', linestyle=':', label='t_critical')
plt.title('t-distribution: Critical and Calculated Values')
plt.legend()
plt.show()

# %%

# Visualization 4: Confidence Interval for mean
plt.figure(figsize=(6, 2))
plt.errorbar(0, score.mean(), 
             yerr=[[score.mean() - confidence_interval[0]], [confidence_interval[1] - score.mean()]], 
             fmt='o', color='blue', capsize=10, label='95% CI')
plt.scatter(0, mu, color='red', zorder=5, label='Mu (8)')
plt.xlim(-1, 1)
plt.xticks([])
plt.title('Confidence Interval for Mu')
plt.legend()
plt.show()

# %%

# T-test for 1 sample
#-- H0: mu = 8
#-- H1: mu != 8
#-- alpha = 0.05
t_test = stats.ttest_1samp(score, mu)
print(f"T-test for 1 sample:")
print(f"t = {t_test.statistic:.2f}, p-value = {t_test.pvalue:.2f}")

if t_test.pvalue >= alpha:
    print(f"Do not reject H0: The mean is equal to {mu}")
else:
    print(f"Reject H0: The mean is {H1} than {mu}")
    

