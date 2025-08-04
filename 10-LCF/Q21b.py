
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
import matplotlib.colors as mcolors
import seaborn as sns #- data visualization 
from tabulate import tabulate # pretty tables

#- Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error #- error function
from statsmodels.stats.outliers_influence import variance_inflation_factor #- VIF
import scipy.stats as stats #- data modeling
from scipy.stats import boxcox, boxcox_llf, boxcox_normmax
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
from sklearn.cluster import KMeans #- K-means library
from yellowbrick.cluster import SilhouetteVisualizer #- Silhuette point by point


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
module_path = str(Path("/Users/kel_m/OneDrive/Nerd_Code/modules-python/").resolve())  # Absolute path
sys.path.append(str(module_path))

#- Linear Regression
from linear_regression_functions import *

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
#- Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# %%
#- Check if User_ID is unique
unique_user_ids = df['User_ID'].nunique()
total_rows = len(df)
print(f"Unique User_ID: {unique_user_ids} (Total rows: {total_rows})")
print("Is User_Id unique for each row?", unique_user_ids == total_rows)

# %%
#- Remove User_ID as variable
df1 = df.copy()
df1 = df1.drop(['User_ID'], axis=1)

# %%
#- Count Gender
gender_counts = df1['Gender'].value_counts()
print("Count of Gender:")
print(gender_counts)

df1_male = df1[df1['Gender'] == 'male']
df1_female = df1[df1['Gender'] == 'female']

# %%
## DATA VISUALIZATION
df_plot = df1.copy()
variable = 'Calories'  # Variable to plot

#- Histogram & Boxplot
fig, (ax1, ax2) = plt.subplots(2, 1, 
                               figsize=(6, 4), 
                               sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})

# Histogram - top subplot
sns.histplot(df_plot[variable], bins=30, kde=True, ax=ax1)
ax1.set_title(f'Histogram of {variable}')
ax1.set_xlabel('')  # Remove x-label to avoid duplication (shared x-axis)
ax1.set_ylabel('Frequency')

# Box plot - bottom subplot
sns.boxplot(data=df_plot, x=variable, ax=ax2)
ax2.set_title(f'Box Plot of {variable}')
ax2.set_xlabel(variable)
ax2.set_ylabel('')

# Plot the mean value as a red 'x' marker
mean_val = df_plot[variable].mean()
ax2.scatter(mean_val, 0, color='red', marker='x', s=100, zorder=5, label='Mean')
# Place the legend outside the boxplot
# ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout() # Adjust layout to prevent overlap
plt.show() # Show the plot

# %%
#- Descriptive Statistics Summary
df1.describe()
# df1_male.describe()
# df1_female.describe()

# %%
df2 = df1.copy()
#- Convert 'Gender' to dummy variable: 1 for Male, 0 for Female
df2['Male'] = (df2['Gender'] == 'male').astype(int)

#- BMI (Body Mass Index)
df2['BMI'] = df2['Weight'] / (df2['Height']/100)**2  # Height in cm → m

#- Caloric Efficiency: Measures how long someone sustains activity per heart rate unit.
df2['caloric_eff'] = df2['Duration'] / df2['Heart_Rate']

#- Thermal Stress: Captures combined effect of exertion time and body heat.
df2['thermal_stress'] =  df2['Body_Temp'] *  df2['Duration']


# %%
remove_var = ['Gender', 
              'Body_Temp', 
              'Duration', 
              'Weight',
            #   'Height',
              'Heart_Rate',
              'caloric_eff',
              'Calories'
              ]
#- Final dataset
df_all = df2.copy()
df_all = df_all.drop(remove_var, axis=1)
df_male = df_all[df_all['Male'] == 1]
df_female = df_all[df_all['Male'] == 0]

# %%
## Correlation Matrix
#- Plot correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_all.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# %%
python_chart_correlation(df_all,figsize=(12, 8))

# %%
#- Identify high correlations (absolute value > 0.7)
high_corr = df_all.corr().abs().stack().reset_index()
high_corr = high_corr[high_corr[0] > 0.7]
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
print("\nHighly Correlated Features:")
print(high_corr)

# %%
# MODEL: CLUSTERING
###########################
df_cluster = df_all.copy()
df_cluster = df_cluster.drop(['Male'], axis=1)

df_cluster_std = df_cluster.copy()
df_cluster_std = same_scale(df_cluster_std)

# %%
## Clustering - Non-Hierarchical Clustering
#- Elbow for k-means method
sum_list = [KMeans(n_clusters=cluster, random_state=1985).fit(df_cluster_std).inertia_ \
              for cluster in range(1,11)  ]


plt.figure(figsize=(8, 5))
plt.plot(range(1,11), sum_list, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#%%
#- Silhuette for k-means method
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=1985)
    cluster_labels = kmeans.fit_predict(df_cluster_std)
    silhouette_avg = silhouette_score(df_cluster_std, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotar o gráfico de Silhouette
plt.figure(figsize=(8, 5))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Average Score')
plt.show()

# %%
### Number of groups chosen
#- Adjust K-means to that number of grups
kmeans = KMeans(n_clusters=5, random_state=1985)
model = kmeans.fit(df_cluster_std)

#- Visualize Silhouete groups
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

#- Plot the chart
visualizer.fit(df_cluster_std)  
visualizer.show()    

# %%
#- Add group number on both standardized and original datasets
df_cluster_std['cluster'] = kmeans.labels_
df_cluster['cluster'] = kmeans.labels_
df_cluster
# df_all['cluster'] = kmeans.labels_

# %%
### Descriptive Statistics Table

grouped = df_cluster.groupby('cluster').describe()

grouped_transposed = grouped.stack(level=0).reset_index()
#- Changed level_1 para variavels
grouped_transposed = grouped_transposed.rename(columns={'level_1': 'variable'})
#- Change the order of the metrics
ordered_columns = ['cluster', 'variable', 'count', 'min', '25%', '50%', '75%', 'max', 'mean', 'std']
grouped_transposed = grouped_transposed[ordered_columns]
grouped_transposed

#- Added color to the rows for easy visualization of groups

#- Function to create a color mapping for clusters
def get_cluster_colors(clusters):
    unique_clusters = clusters['cluster'].unique()
    #- Use seaborn's 'husl' palette for a balanced, varied set of colors
    palette = sns.color_palette("husl", n_colors=len(unique_clusters))
    #- Convert RGB tuples to hex codes
    hex_palette = [mcolors.rgb2hex(rgb) for rgb in palette]
    #- Map each cluster to a color in sequential order
    return {cluster: hex_palette[i] for i, cluster in enumerate(unique_clusters)}

#- Function to color rows based on cluster
def color_rows(row, color_map):
    return ['background-color: {}'.format(color_map[row['cluster']])] * len(row)

#- Assuming grouped_transposed is your DataFrame
#- Generate color mapping for clusters
color_map = get_cluster_colors(grouped_transposed)

#- Apply the coloring to the DataFrame
styled_table = grouped_transposed.style.apply(color_rows, color_map=color_map, axis=1)

#- Display the styled table
styled_table


# %%
### Data Visualization
#- Data
df_chart = df_cluster.copy()
vars_analyzed = ['Age', 'Height', 'BMI', 'thermal_stress']

#- Define how many charts per row
chart_per_row = 2
#- Calculate the number of rows needed
nbr_rows = -(-len(vars_analyzed) // chart_per_row)  
#- Cheate figure subplots
fig, axes = plt.subplots(nbr_rows, chart_per_row, figsize=(12, 4 * nbr_rows))
#- Adjust subplots layout
fig.tight_layout(pad=5.0)

for i, var in enumerate(vars_analyzed):
    #- Select the correct position of the subplot
    ax = axes[i // chart_per_row, i % chart_per_row]
    
    #- Create a neew column "Group" that keep all clusters and add "General" just for the plots
    df_chart['Group'] = df_chart['cluster'].astype(str)
    
    #- Create a temporary copy of the "General" group that has all observations
    df_general = df_chart.copy()
    df_general['Group'] = 'General'
    
    #- Concatenate General and Cluster so the chats are separated
    df_combined = pd.concat([df_chart, df_general])
    
    #- Define the order of the Groups, putting "General" first
    order = ['General'] + sorted(df_chart['cluster'].unique().astype(str))
    
    #- Plot boxplot general and by cluster
    sns.boxplot(ax=ax, 
                x='Group', 
                y=var, 
                data=df_combined, 
                hue='Group',
                palette='Set1', 
                order=order,
                legend=False)
    
    #- Adjust titles and labels
    ax.set_title(f'Boxplot of {var}: General and by Cluster')
    ax.set_xlabel('Group')
    ax.set_ylabel(var)

# %%
# MODEL: LINEAR REGRESSION
###########################
remove_var = ['Gender', 
              'Body_Temp', 
            #   'Duration', 
              'Weight',
            #   'Height',
              'Heart_Rate',
              'caloric_eff',
              'thermal_stress',
            #   'Calories'
              ]
#- Final dataset
df_all_rl = df2.copy()
df_all_rl = df_all_rl.drop(remove_var, axis=1)

df_all_rl['cluster'] = kmeans.labels_
df_all_rl.info()

# %%
#- Count Clusters
cluster_counts = df_all_rl['cluster'].value_counts()
print("Count of cluster:")
print(cluster_counts)

# %%
## Dummys
#- Generic dummy creation: choose reference cluster and auto order
#- Define mapping for cluster labels
cluster_label_map = {
    0: '0_Veteran_Athletes',
    1: '1_Young_Sedentary',
    2: '2_Elite_Athletes',
    3: '3_Short_Veterans',
    4: '4_Light_Fitness'
}

#- Choose the reference cluster
reference_cluster = 4

#- Get all cluster numbers present, put reference first, then the rest
all_clusters = sorted(df_all_rl['cluster'].unique())
ordered_clusters = [reference_cluster] + [c for c in all_clusters if c != reference_cluster]

#- Create ordered group labels
ordered_group_labels = [cluster_label_map[c] for c in ordered_clusters]

#- Map cluster numbers to group names
df_all_rl['Group'] = df_all_rl['cluster'].map(cluster_label_map)

#- Set Group as categorical with desired order
df_all_rl['Group'] = pd.Categorical(df_all_rl['Group'], categories=ordered_group_labels, ordered=True)

#- Create dummies, dropping the reference group
df_all_adj = pd.get_dummies(df_all_rl, columns=['Group'], drop_first=True).astype('int64')
df_all_adj = df_all_adj.drop(['cluster'], axis=1)

df_all_adj.info()

# %%
## Correlation Matrix
#- Plot correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_all_adj.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# %%
## VIF
#- Calculate VIF for each feature
var_Y = 'Calories'
X_vif = df_all_adj.drop(var_Y, axis=1).select_dtypes(include=['int64','float64'])
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values.astype('float'), i) 
                  for i in range(X_vif.shape[1])]

print("Variance Inflation Factors:")
print(vif_data.sort_values("VIF", ascending=False))

#- Remove features with VIF > 5
high_vif = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
df_filtered = df_all_adj.drop(high_vif, axis=1)
print(f"\nRemoved features due to high VIF: {high_vif}")

# %%
## MODEL
X = df_all_adj.drop(['Calories', 
                     'Group_0_Veteran_Athletes',
                     'Group_1_Young_Sedentary',
                     'Group_2_Elite_Athletes',
                     'Group_3_Short_Veterans',], axis=1).astype('float64')
X = sm.add_constant(X) 
y = df_all_adj['Calories']

model = sm.OLS(y, X).fit()
print(model.summary())

# %%
plot_regression_diagnostics(model, 
                            figsize=(12, 8), 
                            lowess_frac=0.4, 
                            point_size=20, 
                            alpha=0.6, 
                            cook_thresholds=(0.5, 1));


# %%
step_columns = step(var_dependent='Calories', 
                    var_independent= df_all_adj.drop(['Calories',
                                                      'Group_0_Veteran_Athletes',
                                                      'Group_1_Young_Sedentary',
                                                      'Group_2_Elite_Athletes', 
                                                      'Group_3_Short_Veterans',
                                                      ], axis = 1).columns.to_list(), 
                    dataset = df_all_adj, 
                    method = 'both' ,
                    metric='aic', 
                    signif=0.05)
step_columns

# %%
### Running the model

#- Get the variable list (pvalue)
# X_stepw_p = df_all_adj[step_columns['var'].to_list()] 
# #- Add the intercept
# X_stepw_p = sm.add_constant(X_stepw_p)
# #- Run the model
# stepw_p = sm.OLS(y, X_stepw_p).fit()
# #- Summary
# print(stepw_p.summary()) 
# #- Generate predicted
# pred_stepw_p = stepw_p.predict(X_stepw_p)

# %%
#- Get the variable list
X_stepw = df_all_adj[ step_columns['var'].to_list()[0] ] 
#- Add the intercept
X_stepw = sm.add_constant(X_stepw)
#- Run the model
stepw = sm.OLS(y, X_stepw).fit()
#- Summary
print(stepw.summary()) 
#- Generate predicted
pred_stepw = stepw.predict(X_stepw)


# %%
### Residual Analysis
plot_regression_diagnostics(stepw, 
                            figsize=(12, 8), 
                            lowess_frac=0.4, 
                            point_size=20, 
                            alpha=0.6, 
                            cook_thresholds=(0.5, 1))

# %%
#### Box-Cox transformation
df_transf = df_all_adj.copy()

#- Apply Box-Cox transformation to the 'mpg' variable
df_transf['Calories_boxcox'], lambda_value = boxcox(df_transf['Calories'])

#- Print the lambda value
print(f"Lambda value for Box-Cox transformation: {lambda_value}")

# %%
# Get the data (must be positive)
y = df_all_adj['Calories'].values

# Find the lambda that maximizes the log-likelihood
lambda_mle = boxcox_normmax(y, method='mle')

# Compute the log-likelihood for a range of lambda values
lambdas = np.linspace(-2, 2, 100)
llf = [boxcox_llf(lmb, y) for lmb in lambdas]

# Find the confidence interval (approx 95% CI: drop in log-likelihood of 1.92 from max)
llf_max = np.max(llf)

# Find the confidence interval (approx 95% CI: drop in log-likelihood of 1.92 from max)
ci_mask = np.array(llf) > llf_max - 1.92
ci_lambdas = lambdas[ci_mask]
ci_low = ci_lambdas[0]
ci_high = ci_lambdas[-1]
print(f"Approximate 95% confidence interval for lambda: [{ci_low:.3f}, {ci_high:.3f}]")

import matplotlib.pyplot as plt
plt.plot(lambdas, llf, label='Log-likelihood')
plt.axvline(lambda_mle, color='red', linestyle='--', label=f'Lambda MLE: {lambda_mle:.2f}')
plt.axvspan(ci_low, ci_high, color='gray', alpha=0.3, label=f'~95% CI: [{ci_low:.2f}, {ci_high:.2f}]')
plt.annotate(f'CI Low\n{ci_low:.2f}', xy=(ci_low, llf_max-1.5), xytext=(ci_low-0.5, llf_max-3),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)
plt.annotate(f'CI High\n{ci_high:.2f}', xy=(ci_high, llf_max-1.5), xytext=(ci_high+0.2, llf_max-3),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)
plt.xlabel('Lambda')
plt.ylabel('Log-likelihood')
plt.title('Box-Cox Lambda Confidence Interval')
plt.legend()
plt.show()

# %%
step_columns = step(var_dependent='Calories_boxcox', 
                    var_independent= df_transf.drop(['Calories', 
                                                     'Group_0_Veteran_Athletes',
                                                      'Group_1_Young_Sedentary',
                                                      'Group_2_Elite_Athletes', 
                                                      'Group_3_Short_Veterans',
                                                     'Calories_boxcox'], axis = 1).columns.to_list(), 
                    dataset = df_transf, 
                    method = 'both' ,
                    metric='aic', 
                    signif=0.05)
step_columns


# %%
X = df_transf.drop(['Calories', 
                     'Group_0_Veteran_Athletes',
                     'Group_1_Young_Sedentary',
                     'Group_2_Elite_Athletes',
                     'Group_3_Short_Veterans',
                     'Calories_boxcox'], axis=1).astype('float64')
X = sm.add_constant(X) 
y = df_transf['Calories_boxcox']

model = sm.OLS(y, X).fit()
print(model.summary())




# %%
### Residual Analysis
plot_regression_diagnostics(model, 
                            figsize=(12, 8), 
                            lowess_frac=0.4, 
                            point_size=20, 
                            alpha=0.6, 
                            cook_thresholds=(0.5, 1));

# %%

# %%
alpha = 0.05
# Perform the Shapiro-wilk test
shapiro_test = stats.shapiro(model.resid)
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Interpret the result
print(f"In this case, the p-value is {shapiro_test.pvalue:.4f} and with an alpha of {alpha}, we can conclude that:")

if shapiro_test.pvalue < alpha:
    print("Reject the null hypothesis that the residuals are normally distributed.")
else:
    print("Fail to reject the null hypothesis that the residuals are normally distributed.")

# %%
### Statistical Tests for Heteroscedasticity
#- Breusch-Pagan Test (Best for Linear Models)
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan Test:")
print(f"Lagrange Multiplier: {bp_test[0]:.3f}, p-value: {bp_test[1]:.3f}")

# Interpret the result
print(f"In this case, the p-value is {bp_test[1]:.4f} and with an alpha of {alpha}, we can conclude that:")

if bp_test[1] < alpha:
    print("Reject the null hypothesis that the residuals are distributed with equal variance\n(heteroscedasticity is present).")
else:
    print("Fail to reject the null hypothesis that the residuals are distributed with equal variance\n(homoscedasticity is present).")


# %%
