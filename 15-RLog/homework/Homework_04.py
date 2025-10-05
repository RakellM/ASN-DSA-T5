#%%
# Homework 04 - Q10

## Library
import pandas as pd

from important_functions import *

# %%

df = pd.read_csv("../data/bank.csv", sep=";")

df.info()

# %%
df.head()

# %%

univariate_numeric_variable(df, 'age')

# %%


# %%
univariate_categorical_variable(df)