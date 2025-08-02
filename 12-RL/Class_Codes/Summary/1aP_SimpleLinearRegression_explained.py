# SIMPLE LINEAR REGRESSION
##########################################################################

# %%
## LIBRARY
##################################
import pandas as pd #- data manipulation
import statsmodels.api as sm #- logistic regression
import numpy as np #- mathematical calculations
from matplotlib import pyplot as plt #- data visualization 
import seaborn as sns #- data visualization 
from tabulate import tabulate # pretty tables
from sklearn.metrics import mean_squared_error, mean_absolute_error #- error function
from statsmodels.stats.outliers_influence import variance_inflation_factor #- VIF
import scipy.stats as stats #- data modeling
import pylab #- QQplot
from statsmodels.stats.diagnostic import het_breuschpagan #- Breusch-Pagan variance test
from scipy.stats import shapiro #- Shapiro-Wilk normality test
from statsmodels.nonparametric.smoothers_lowess import lowess #- Lowess for Resudual plot

# %%
### Discussion: Selection mothods Backward, Forward and Stepwise
#- Python does not has the selection methods built in on the most used packages.
#- Solution here is to create these functions.
#- Check file: linear_regression_functions.py
from linear_regression_functions import step, charts_var_num

# %% 
## DATASET
##################################
#- Simple dataset
df = pd.DataFrame({"students": [2,6,8,8,12,16,20,20,22,26],
                    "pizza": [55,105,88,118,117,137,157,169,149,202]
                })
df.head()


# %%
## EDA
##################################
### Summary of Dependent Variable (Y)
df["pizza"].describe()

# %%
#- Histogram
# sns.histplot(data=df, x="pizza")
sns.histplot(data=df, x="pizza", binrange=[50,250], binwidth=50 )

# %%
#- Box-Plot
sns.boxplot(df["pizza"])

# %%
#- Scatter Plot 
df.plot(kind = "scatter", x= "students", y="pizza")

# %%
## MODEL
##################################
# Independent Variables (Matrix)
X = df["students"] 
# Dependent Variable (vector)
y = df["pizza"] 
# Add the constant column on the X matrix
X = sm.add_constant(X) 
# Adjusting the model
simple_reg = sm.OLS(y, X).fit()

# Summary
print(simple_reg.summary())

# %%
## RESIDUAL ANALYSIS
##################################
# df['fitted'] = simple_reg.predict(X)
# df['residuals'] = df['pizza'] - df['fitted']

df['fitted'] = simple_reg.fittedvalues
df['residuals'] = simple_reg.resid

# %%
#- Create figure and axes
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle('Regression Diagnostics (R-style)', y=1.02)

# --- Plot 1: Residuals vs Fitted (Top Left) ---
fitted = simple_reg.fittedvalues
residuals = simple_reg.resid
lowess_line = lowess(residuals, fitted, frac=0.33)[:, 1]

axes[0,0].scatter(fitted, residuals, alpha=0.6, edgecolors='k')
axes[0,0].axhline(y=0, color='gray', linestyle='--')
axes[0,0].plot(fitted, lowess_line, 'r', linewidth=2)
axes[0,0].set_xlabel("Fitted Values")
axes[0,0].set_ylabel("Residuals")
axes[0,0].set_title("Residuals vs Fitted")

# Optional: Add mean residual line
mean_resid = np.mean(residuals)
axes[0,0].axhline(y=mean_resid, color='blue', linestyle=':', linewidth=1)

# --- Plot 2: Q-Q Plot (Top Right) ---
sm.qqplot(residuals, line='45', fit=True, ax=axes[0,1])
axes[0,1].set_title("Normal Q-Q")

# --- Plot 3: Scale-Location (Bottom Left) ---
sm.graphics.plot_leverage_resid2(simple_reg, ax=axes[1,0])
axes[1,0].set_title("Scale-Location")

# --- Plot 4: Residuals vs Leverage (Bottom Right) ---
sm.graphics.influence_plot(simple_reg, ax=axes[1,1])
axes[1,1].set_title("Residuals vs Leverage")

#- Adjust layout
plt.tight_layout()
plt.show()
# %%

# %%
