#- linear_regression_functions.py

# %%
# LIBRARY
##################################
import pandas as pd #- data manipulation
import numpy as np #- math
import scipy.stats as stats #- data modeling
import statsmodels.api as sm #- logistic regression
from matplotlib import pyplot as plt #- data visualization
import seaborn as sns #- data visualization
from statsmodels.stats.diagnostic import het_breuschpagan #- Breusch-Pagan variance test
from statsmodels.nonparametric.smoothers_lowess import lowess #- Lowess for Resudual plot

# %%
## FUNCTION: Forward p-value
def select_pvalue_forward(var_dependent, var_independent, dataset, signif):
 
    """   
    This function performs forward stepwise selection based on the p-value 
    of the independent variables.
    At each step, it adds the independent variable with the lowest p-value 
    to the model, provided that the p-value is smaller than the specified 
    significance level.
    
    Parameters:
      var_dependent (str): Name of the dependant variable.
      var_independent (list): List of independent variables to be evaluated.
      dataset (pd.DataFrame): Dataset containing both dependent and independent variables.
      signif (float): Significance level for variable inclusion (e.g., 0.05).
    
    Returns: 
        pd.DataFrame: DataFrame containing the selected variable an their respective p-values.
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> pvalue_columns = select_pvalue_forward(var_dependent='BodyFat', var_independent=df.drop('BodyFat', axis = 1).columns.to_list(), dataset=df, signif=0.05)
            >>> pvalue_columns
    
    Created by Mateus Rocha - ASN.Rocks team
    """
    
    predicted = []
    pvalue_predicted = []
    Y = dataset[var_dependent]
    while True and var_independent != [] :
        list_pvalue = []
        list_variable = []
        for var in var_independent:
            X = sm.add_constant(dataset[ [var] +  predicted ])
            
            model = sm.OLS(Y,X).fit()
            
            if( predicted == []):
    
                pvalue = model.pvalues[1]
                variable = model.pvalues.index[1]
            
            else:
                pvalue = model.pvalues.drop(predicted)[1]
                variable = model.pvalues.drop(predicted).index[1]
                
            list_pvalue.append(pvalue)
            list_variable.append(variable)          
        
        if( list_pvalue[ np.argmin(list_pvalue) ] < signif ):
            predicted.append( list_variable[np.argmin(list_pvalue)] )
            pvalue_predicted.append(list_pvalue[ np.argmin(list_pvalue) ])
            var_independent.remove( list_variable[ np.argmin(list_pvalue)] )
        else:
            break
    final_info = pd.DataFrame({ 'var': predicted, 'pvalue': pvalue_predicted})
    return final_info

# %%
## FUNCTION: Forward aic
def select_aic_forward(var_dependent, var_independent, dataset):

    """   
    This function performs forward stepwise selection based on the Akaike 
    Information Criterion (AIC).
    At each step, it adds the independent variable that minimizes the AIC 
    to the model.
    
    Parameters:
      var_dependent (str): Name of the dependent variable.
      var_independent (list): List of independent variables to be evaluated.
      dataset (pd.DataFrame): Dataset containing both dependent and independent variables.
    
    Returns: 
        pd.DataFrame: DataFrame containing the selected variable combinations 
        and their respective AICs, sorted from the lowest to the highest AIC.
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> aicforward_columns = select_aic_forward(var_dependent='BodyFat', var_independent=df.drop('BodyFat', axis = 1).columns.to_list(), dataset=df)
            >>> aicforward_columns
    
    Created by Mateus Rocha - ASN.Rocks team
    """
    
    predicted = []
    aic_predicted = []
    Y = dataset[var_dependent]
    final_list = []
    aic_best = float('inf')
    
    while True and var_independent != []:
        list_aic = []
        list_variable = []
        list_models =[]
        if(var_independent == []):
            break
        for var in var_independent:
            X = sm.add_constant(dataset[ [var] +  predicted ])
            aic = sm.OLS(Y,X).fit().aic
            variable = var
                
            list_aic.append(aic)
            
            list_variable.append(var)
            
            list_models.append( [var] +  predicted )
            
        if( list_aic[ np.argmin(list_aic) ] < aic_best ):
            
            final_list.append(list_models[ np.argmin(list_aic)]  )
            
            predicted.append( list_variable[np.argmin(list_aic)] )
            
            aic_predicted.append(list_aic[ np.argmin(list_aic) ])
            
            var_independent.remove( list_variable[ np.argmin(list_aic)] )
            
            aic_best = list_aic[ np.argmin(list_aic) ] 
            
        else:
            break
        
    final_info = pd.DataFrame({ 'var': final_list, 'aic': aic_predicted}).sort_values(by = 'aic')
    return final_info

# %%
## FUNCTION: Forward bic
def select_bic_forward(var_dependent, var_independent, dataset):

    """   
    This function performs forward stepwise selection based on the Bayesian Information Criterion (BIC).
    At each step, it adds the independent variable that minimizes the BIC to the model.
    
    Parameters:
      var_dependent (str): Name of the dependent variable.
      var_independent (list): List of independent variables to be evaluated.
      dataset (pd.DataFrame): Dataset containing both dependent and independent variables.
    
    Returns: 
        pd.DataFrame: DataFrame containing the selected variable combinations and their respective BIC scores,
        sorted from lowest to highest BIC.
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> bicforward_columns = select_bic_forward(var_dependent='BodyFat', var_independent=df.drop('BodyFat', axis = 1).columns.to_list(), dataset=df)
            >>> bicforward_columns
    
    Created by Mateus Rocha - ASN.Rocks team
    """
    
    predicted = []
    bic_predicted = []
    Y = dataset[var_dependent]
    final_list = []
    bic_best = float('inf')
    
    while True and var_independent != []:
        list_bic = []
        list_variable = []
        list_models =[]
        if(var_independent == []):
            break
        for var in var_independent:
            X = sm.add_constant(dataset[ [var] +  predicted ])
            bic = sm.OLS(Y,X).fit().bic
            variable = var
                
            list_bic.append(bic)
            
            list_variable.append(var)
            
            list_models.append( [var] +  predicted )
            
        if( list_bic[ np.argmin(list_bic) ] < bic_best ):
            
            final_list.append(list_models[ np.argmin(list_bic)]  )
            
            predicted.append( list_variable[np.argmin(list_bic)] )
            
            bic_predicted.append(list_bic[ np.argmin(list_bic) ])
            
            var_independent.remove( list_variable[ np.argmin(list_bic)] )
            
            aic_best = list_bic[ np.argmin(list_bic) ] 
            
        else:
            break
        
    final_info = pd.DataFrame({ 'var': final_list, 'bic': bic_predicted}).sort_values(by = 'bic')
    return final_info

#%%
## FUNCTION: backward p-value
def select_pvalue_backward(var_dependent, var_independent, dataset, signif):

    """   
    Performs backward stepwise selection based on p-values of independent variables.
    Iteratively removes the variable with the highest p-value from the model, 
    provided it exceeds the specified significance level.
    
    Parameters:
      var_dependent (str): Name of the dependent variable.
      var_independent (list): List of independent variables to evaluate.
      dataset (pd.DataFrame): Dataset containing both dependent and independent variables.
      signif (float): Significance threshold for variable retention (e.g., 0.05).
      
    Returns: 
        pd.DataFrame: Variables remaining after backward selection with their statistics.
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> pvaluebackward_columns = select_pvalue_backward(var_dependent='BodyFat', var_independent=df.drop('BodyFat', axis = 1).columns.to_list(), signif = 0.05 , dataset=df)
            >>> pvaluebackward_columns
    
    Created by Mateus Rocha - ASN.Rocks team
    """
    
    Y = dataset[var_dependent]
    
    while True and var_independent != []:
        
        X_general = sm.add_constant(dataset[var_independent])
        
        model = sm.OLS(Y,X_general).fit()
        
        pvalue_general = model.pvalues
        
        variable_general = model.pvalues.index
        
        if(pvalue_general[ np.argmax(pvalue_general) ] > signif ):
            var_independent.remove( variable_general[ np.argmax(pvalue_general) ] )
        else:
            break
    
    
    final_info = pd.DataFrame({ 'var': var_independent})
    return final_info

# %%
## FUNCTION: Backward aic
def selecionar_aic_backward(var_dependent, var_independent, dataset):

    """   
    Performs backward stepwise variable selection using the Akaike Information Criterion (AIC).
    Iteratively removes the variable whose exclusion results in the lowest AIC improvement.
    
    Parameters:
      var_dependent (str): Name of the target response variable.
      var_independent (list): List of independent variables to evaluate.
      dataset (pd.DataFrame): Dataset containing both dependent and independent variables.
    
    Returns: 
        pd.DataFrame: Optimal variable combinations with their AIC values,
        sorted by ascending AIC (best models first).
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> aicbackward_columns = selecionar_aic_backward(var_dependent='BodyFat', var_independent=df.drop('BodyFat', axis = 1).columns.to_list(), dataset=df)
            >>> aicbackward_columns
    
    Created by Mateus Rocha - ASN.Rocks team
    """
    
    Y = dataset[var_dependent]
    
    predicted_finais = []
    
    aic_final = []
    
    while True and var_independent != []:
        
        list_aic = []
        list_predicted = []

        X_general = sm.add_constant(dataset[var_independent])
        
        aic_general = sm.OLS(Y,X_general).fit().aic
    
        aic_final.append(aic_general)
        
        predicted_finais.append(dataset[var_independent].columns.to_list())
        
        for var in var_independent:
            
            list_variaveis = var_independent.copy()
            list_variaveis.remove(var)
            
            X = sm.add_constant(dataset[ list_variaveis ])
            aic = sm.OLS(Y,X).fit().aic    
            
            list_aic.append(aic)
            
            list_predicted.append(var)
            
        if(list_aic[ np.argmin(list_aic) ] < aic_general ):
            var_independent.remove( list_predicted[ np.argmin(list_aic) ] )
            
        else:
            break
    
    
    final_info = pd.DataFrame({ 'var': predicted_finais, 'aic':aic_final }).sort_values(by = 'aic')
    return final_info

# %%
## FUNCTION: Backward bic
def select_bic_backward(var_dependent, var_independent, dataset):
    
    """   
    Performs backward stepwise selection using the Bayesian Information Criterion (BIC).
    Iteratively removes the least significant variable (highest BIC impact) at each step.
    
    Parameters:
      var_dependent (str): Name of the target response variable.
      var_independent (list): List of independent variables to evaluate.
      dataset (pd.DataFrame): Dataset containing both dependent and independent variables.
    
    Returns: 
        pd.DataFrame: Optimal variable combinations with their BIC scores,
        sorted in ascending order (best models first)
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> bicbackward_columns = select_bic_backward(var_dependent='BodyFat', var_independent=df.drop('BodyFat', axis = 1).columns.to_list(), dataset=df)
            >>> bicbackward_columns
    
    Created by Mateus Rocha - ASN.Rocks team
    """
    
    Y = dataset[var_dependent]
    
    predicted_finais = []
    
    bic_final = []
    
    while True and var_independent != []:
        
        list_bic = []
        list_predicted = []

        X_general = sm.add_constant(dataset[var_independent])
        
        bic_general = sm.OLS(Y,X_general).fit().bic
    
        bic_final.append(bic_general)
        
        predicted_finais.append(dataset[var_independent].columns.to_list())
        
        for var in var_independent:
            
            list_variaveis = var_independent.copy()
            list_variaveis.remove(var)
            
            X = sm.add_constant(dataset[ list_variaveis ])
            bic = sm.OLS(Y,X).fit().bic    
            
            list_bic.append(bic)
            
            list_predicted.append(var)
            
        if(list_bic[ np.argmin(list_bic) ] < bic_general ):
            var_independent.remove( list_predicted[ np.argmin(list_bic) ] )
            
        else:
            break
    
    
    final_info = pd.DataFrame({ 'var': predicted_finais, 'bic':bic_final }).sort_values(by = 'bic')
    return final_info

# %%
## FUNCTION: Stepwise
def stepwise( var_dependent , var_independent , dataset, metric, signif = 0.05, epsilon = 0.0001):
    
    """   
    Performs hybrid stepwise variable selection combining forward and backward methods
    using a specified metric (AIC, BIC, or p-value). 
    The algorithm first applies forward selection with the chosen metric, then backward 
    selection, iterating until the metric difference falls below a tolerance threshold (epsilon).
    
    Parameters:
      var_dependent (str): Name of the target response variable.
      var_independent (list): List of independent variables to evaluate.
      dataset (pd.DataFrame): Dataset containing both dependent and independent variables.
      metric (str):Selection criterion ('aic', 'bic', or 'pvalue').
      signif (float): Significance level for p-value selection (default=0.05).
      epsilon (float): Minimum metric difference threshold for convergence (default=0.0001).

    Returns: 
        Optimal variable subset with selection metrics and statistics.
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> stepwise_columns = stepwise(var_dependent='BodyFat', var_independent=df.drop('BodyFat', axis = 1).columns.to_list(), dataset = df , metric='aic', signif=0.05)
            >>> stepwise_columns
    
    Created by Mateus Rocha - ASN.Rocks team
    """

    
    list_var = var_independent
    
    metric_forward = 0
    
    metric_backward = 0
    
    while True:
    
        if(metric == 'aic'):
            result = select_aic_forward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset)

            if (len(result) == 1):
                return result
            
            result_final = selecionar_aic_backward(var_dependent = var_dependent, var_independent = result['var'].to_list()[0], dataset = dataset)

            if(len(result_final) == 1):
                return result_final

            metric_forward = result['aic'].to_list()[0]

            metric_backward = result_final['aic'].to_list()[0]


        elif(metric == 'bic'):
            result = select_bic_forward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset)

            if (len(result) == 1):
                return result

            result_final = select_bic_backward(var_dependent = var_dependent, var_independent = result['var'].to_list()[0], dataset = dataset)

            if(len(result_final) == 1):
                return result_final

            metric_forward = result['bic'].to_list()[0]

            metric_backward = result_final['bic'].to_list()[0]

        elif(metric == 'pvalue'):
            result = select_pvalue_forward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset, signif = signif)

            if (len(result) == 1):
                return result

            result_final = select_pvalue_backward(var_dependent = var_dependent, var_independent = result['var'].to_list(), dataset = dataset, signif = signif)

            if(len(result_final) == 1):
                return result_final

            return result_final

        if( abs(metric_forward - metric_backward) < epsilon ):
            break
        else:
            var_independent = set(result_final['var'].to_list() + list_var)    

# %%
## FUNCTION: Step
def step( var_dependent , var_independent , dataset, method, metric, signif = 0.05):
        
    """   
    Performs variable selection using forward, backward, or hybrid stepwise methods 
    cbased on a specified criterion (AIC, BIC, or p-value).
    The user can select both the method (forward, backward, or both) and the metric 
    for variable inclusion/exclusion.
    
    Parameters:
      var_dependent (str): Name of the target response variable.
      var_independent (list): List of independent variables to evaluate.
      base (pd.DataFrame): Dataset containing both dependent and independent variables.
      metric (str): Selection criterion ('aic', 'bic', or 'pvalue').
      method (str): Selection approach ('forward', 'backward', or 'both').
      signif (float): Significance level for p-value threshold (default=0.05).

    Returns: 
        Optimal variable subset with selection metrics and statistics.
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> step_columns = step(var_dependent='BodyFat', var_independent=df.drop('BodyFat', axis = 1).columns.to_list(), dataset = df, method = 'forward' ,metric='aic', signif=0.05)
            >>> step_columns
    
    Created by Mateus Rocha - ASN.Rocks team
    """
    
    if( method == 'forward' and metric == 'aic' ):
        result = select_aic_forward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset)
    elif(method == 'forward' and metric == 'bic' ):
        result = select_bic_forward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset)
    elif(method == 'forward' and metric == 'pvalue' ):
        result = select_pvalue_forward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset, signif = signif)
    elif( method == 'backward' and metric == 'aic' ):
        result = selecionar_aic_backward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset)
    elif(method == 'backward'and metric == 'bic' ):
        result = select_bic_backward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset)
    elif(method == 'backward' and metric == 'pvalue' ):
        result = select_pvalue_backward(var_dependent = var_dependent, var_independent = var_independent, dataset = dataset, signif = signif)
    elif(method == 'both'):
        result = stepwise( var_dependent = var_dependent , var_independent = var_independent , dataset = dataset, metric = metric, signif = signif)
        
    # -Adjust pandas display settings to prevent truncation of long columns and rows
    pd.set_option('display.max_colwidth', None)  #- prevent truncation of columns
    pd.set_option('display.max_rows', None)  #- show all rows
    
    return result

# %%
## FUNCTION: Combo charts: Histogram / Box-Plot and Violin
def charts_var_num(dataset, variable):

    """   
    This function generates three plots (histogram, boxplot, and violin plot) 
    for a specific variable in a dataset.
    
    Parameters:
        dataset (pd.DataFrame): Input dataset (DataFrame) containing the variables.
        variable (str): Name of the variable to analyze (string).
    
    Returns: 
       Three plots (histogram, boxplot, and violin) displayed side by side for the chosen variable.
        
    Usage example:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> charts_var_num(dataset=df, variable="BodyFat")
    """
    

    # Define figure size generating 3 subplots (1x3)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Histogram
    sns.histplot(data=dataset, x=variable, bins=25, ax=axs[0])
    axs[0].set_title(f"Histogram: {variable}")

    # Boxplot
    sns.boxplot(y=variable, data=dataset, ax=axs[1])
    axs[1].set_title(f"Boxplot: {variable}")

    # Violin plot
    sns.violinplot(y=variable, data=dataset, ax=axs[2])
    axs[2].set_title(f"Violin plot: {variable}")

    # Adjust layout to show titles
    plt.tight_layout()
    plt.show()

# %%
## FUNCTION: Correlation Matrix similar to R
def python_chart_correlation(df, hist_bins=10, figsize=(10, 6), 
                         fontsize=12, color='red', scatter_alpha=0.6):
    """
    Clean correlation plot with:
    - Upper triangle: Correlation values with significance
    - Diagonal: Histograms with KDE
    - Lower triangle: Scatter plots
    
    Parameters:
    df : pandas DataFrame
        Numeric dataframe to plot
    hist_bins : int
        Number of bins for histograms
    figsize : tuple
        Size of the figure
    fontsize : int
        Font size for correlation values
    color : str
        Color for correlation text and significance markers
    scatter_alpha : float
        Transparency for scatter points

    Usage example:
        >>> import pandas as pd
        >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
        >>> python_chart_correlation(df)

    """
    sns.set(style='white', font_scale=1.2)
    
    # Calculate figure size based on number of columns
    n_vars = len(df.columns)
    figsize = (max(10, n_vars*2), (max(6, n_vars*2)))
    
    g = sns.PairGrid(df, diag_sharey=False, height=figsize[0]/n_vars)
    
    # Upper triangle: Correlation values with significance
    g.map_upper(plot_correlation, fontsize=fontsize, color=color)
    
    # Diagonal: Histograms with KDE
    g.map_diag(sns.histplot, bins=hist_bins, kde=True, 
              color='steelblue', edgecolor='white', alpha=0.7)
    
    # Lower triangle: Scatter plots
    g.map_lower(sns.scatterplot, alpha=scatter_alpha, color='steelblue', s=20)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle('Correlation Matrix', y=1.02, fontsize=16)
    plt.show()

def plot_correlation(x, y, fontsize=12, color='red', **kws):
    """Plot correlation coefficient with significance stars"""
    r, p = stats.pearsonr(x, y)
    
    # Create significance indicator
    sig = ''
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    
    # Format the correlation text
    corr_text = f"{r:.2f}\n{sig}"
    
    # Get current axis
    ax = plt.gca()
    
    # Add text with white background for better visibility
    ax.annotate(corr_text, xy=(0.5, 0.5), xycoords='axes fraction',
               ha='center', va='center', fontsize=fontsize, color=color,
               bbox=dict(boxstyle='round,pad=0.3', 
                         fc='white', ec='none', alpha=0.8))

    
# %%
## FUNCTION: Regression Diagnostics
def plot_regression_diagnostics(model, figsize=(12, 8), lowess_frac=0.4, 
                               point_size=20, alpha=0.6, cook_thresholds=(0.5, 1)):
    """
    Create R-style regression diagnostic plots.
    
    Parameters:
    model : statsmodels regression model
        Fitted regression model (OLS, GLM, etc.)
    figsize : tuple
        Figure size (width, height)
    lowess_frac : float
        Fraction of data to use for LOWESS smoothing
    point_size : int
        Size of scatter plot points
    alpha : float
        Transparency of points (0-1)
    cook_thresholds : tuple
        Thresholds for Cook's distance contours
        
    Returns:
    matplotlib Figure object

    Usage example:
        >>> model = sm.OLS(y, X).fit()
        >>> fig = plot_regression_diagnostics(model)
        >>> plt.show()

    """
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Regression Diagnostics (R-style)', y=1.02)
    
    # Get model components
    fitted = model.fittedvalues
    residuals = model.resid
    abs_resid = np.abs(residuals)
    abs_sqrt_resid = np.sqrt(abs_resid)
    student_resid = model.get_influence().resid_studentized_internal
    leverage = model.get_influence().hat_matrix_diag
    cooks = model.get_influence().cooks_distance[0]
    
    # --- Plot 1: Residuals vs Fitted ---
    axes[0,0].scatter(fitted, residuals, alpha=alpha, edgecolors='k', s=point_size)
    axes[0,0].axhline(y=0, color='gray', linestyle='--')
    lowess_line = lowess(residuals, fitted, frac=lowess_frac, it=3)
    axes[0,0].plot(lowess_line[:,0], lowess_line[:,1], 'r', linewidth=1.5)
    axes[0,0].set(xlabel="Fitted Values", ylabel="Residuals", 
                 title="Residuals vs Fitted")
    
    # --- Plot 2: Q-Q Plot ---
    sm.qqplot(residuals, line='45', fit=True, ax=axes[0,1], 
             markersize=point_size/3, alpha=alpha)
    axes[0,1].set_title("Normal Q-Q")
    
    # --- Plot 3: Scale-Location ---
    axes[1,0].scatter(fitted, abs_sqrt_resid, alpha=alpha, 
                     edgecolors='k', s=point_size)
    lowess_scale = lowess(abs_sqrt_resid, fitted, frac=lowess_frac, it=3)
    axes[1,0].plot(lowess_scale[:,0], lowess_scale[:,1], 'r', linewidth=1.5)
    axes[1,0].set(xlabel="Fitted Values", 
                 ylabel="√|Standardized Residuals|", 
                 title="Scale-Location")
    
    # --- Plot 4: Residuals vs Leverage ---
    sc = axes[1,1].scatter(leverage, student_resid, alpha=alpha, s=point_size,
                          c=cooks, cmap='Reds', edgecolors='k')
    
    # Add Cook's distance contours
    x = np.linspace(min(leverage)*0.99, max(leverage)*1.01, 50)
    for c in cook_thresholds:
        y = np.sqrt((c * (1 - x)) / x)
        axes[1,1].plot(x, y, '--', color='gray', alpha=0.5, linewidth=1)
        axes[1,1].plot(x, -y, '--', color='gray', alpha=0.5, linewidth=1)
        axes[1,1].text(x[-1], y[-1], f'Cook\'s {c}', ha='left', va='bottom', 
                       fontsize=8, color='gray')
    
    axes[1,1].axhline(y=0, color='gray', linestyle='--')
    axes[1,1].set(xlabel="Leverage", ylabel="Standardized Residuals",
                 title="Residuals vs Leverage")
    
    # Add colorbar
    smap = plt.cm.ScalarMappable(cmap='Reds', 
                               norm=plt.Normalize(vmin=min(cooks), vmax=max(cooks)))
    smap.set_array([])
    cbar = fig.colorbar(smap, ax=axes[1,1], pad=0.02)
    cbar.set_label("Cook's Distance")
    
    plt.tight_layout()
    return fig

# %%
