
# %%
## LIBRARY
##################################
import os #- system
import sys #- system
from pathlib import Path #- system path to use module

import pandas as pd #- data manipulation
import statsmodels.api as sm #- logistic regression
import numpy as np #- mathematical calculations

from matplotlib import pyplot as plt #- data visualization 
import seaborn as sns #- data visualization 
from tabulate import tabulate # pretty tables

#- Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error #- error function
from statsmodels.stats.outliers_influence import variance_inflation_factor #- VIF
import scipy.stats as stats #- data modeling
import pylab #- QQplot
from statsmodels.stats.diagnostic import het_breuschpagan #- Breusch-Pagan variance test
from scipy.stats import shapiro #- Shapiro-Wilk normality test

#- Cluster
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import tanglegram

import kagglehub #- Kaggle Hub to import datasets

sys.dont_write_bytecode = True  #- Prevents cache creation for modules

# %%
### Discussion: Selection mothods Backward, Forward and Stepwise
#- Python does not has the selection methods built in on the most used packages.
#- Solution here is to create these functions.
#- Check file: linear_regression_functions.py

# 1. Set where you want __pycache__ to be created (current directory)
current_dir = Path(__file__).parent.resolve()  #- Gets your script's directory
os.environ["PYTHONPYCACHEPREFIX"] = str(current_dir / "pycache_temp")

# 2. Now import your module - cache will be created in the specified location
module_path = str(Path("/Users/kel_m/OneDrive/Nerd_Code/modules_python/").resolve())  # Absolute path
sys.path.append(str(module_path))

#- Linear Regression
# from linear_regression_functions import *

#- Hierarchical Cluster
from hierarchical_cluster_functions import *

# %%
## DATASET
##################################
#- Dataset for multiple regression
#- KaggleHub
path = kagglehub.dataset_download(r'ruchikakumbhar/calories-burnt-prediction')
df = pd.read_csv(os.path.join(path, 'calories.csv'))  
print(df.head())

# %%
## PROBLEM
##################################
### About Dataset
#- Features:
#- + User_Id
#- + Gender
#- + Age
#- + Height
#- + Weight
#- + Duration
#- + Heart_rate
#- + Body_temp
#- Target:
#- + Calories

#- Let's imagine we did not have the target variable and let's try a cluster analysis.
#- Imagine that our problem is degine group of people that have similarities in the way their body behave while doing exercises.

df.info()

# %%
## EDA - EXPLORATORY DATA ANALYSIS
################################## 



# %%
