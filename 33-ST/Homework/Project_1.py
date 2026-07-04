# %%
# Time Series Project


# %%
# LIBRARY
import os
import pandas as pd



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
df.head()

# %%
df.info()

# %%
