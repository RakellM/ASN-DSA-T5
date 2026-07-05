# %%
# Time Series Project


# %%
# LIBRARY
# ------------------------------------------------------------------------------
import os
import pandas as pd
from datetime import datetime



# %%
## PATH
project_dir = os.path.join(os.path.expanduser("~"), 
                           "OneDrive", 
                           "Project_Code",
                           "ASN-DSA-T5", 
                           "33-ST",
                           "Homework")


# %%
df = pd.read_csv(os.path.join(project_dir, "data", "Vendas ASN_Dados_Finais.csv"))

print("Dataset Shape:", df.shape)

# %%
df.sample(10)


# %%
# Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------

# raw data type
df.info()

# raw data range values
df.describe()

# %%
# check for missing values
df.isnull().sum()



# %%
# List unique categories
print(df["Departamento"].unique())
print(df["Seção"].unique())
# print(df["Data_new"].unique())


# %%
## Data Cleaning & Preprocessing
# ------------------------------------------

# Variable : Data_new
# Convert date from string to date
df["Date"] = pd.to_datetime(df["Data_new"], errors='coerce')

# Count total missing/invalid dates
missing_count = df["Date"].isna().sum()
print(f"Total missing or errored rows: {missing_count}")

# Filter and display rows where the conversion failed
error_rows = df[df["Date"].isna()]
print(error_rows[["Data_new", "Date"]])

# Check the earliest and latest dates to spot typos
print(df["Date"].describe())


# %%
# Variable: Departamento

