# %%
# Time Series Project


# %%
# LIBRARY
# ------------------------------------------------------------------------------
import os
import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%
# FUNCTION
# ------------------------------------------------------------------------------
def univariate_numeric_variable(data, variable):
    """
    Generates a matrix of charts (2x2) for a numeric continuous variable.

    [1,1] Histogram
    [1,2] Violin Plot
    [2,1] Box plot
    [2,2] Box plot with points overlaid

    Above the charts, shows a table with the variable descriptive statistics.

    Parameters:
        data (pd.DataFrame): Database containing the variable.
        variable (str): Name of the variable to be analysed.

    Returns:
        None

    Usage example:
        >> data = pd.DataFrame({"example_variable": np.random.normal(loc=50, scale=10, size=100)})
        >>univariate_numeric_variable(data, "example_variable")
        
    """
    
    # Calculate descriptive statistics
    desc_stats = data[variable].describe().to_frame().T
    desc_stats = desc_stats.round(4)  # Limit to 4 decimal places

    # Configuration of subplots
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(f"Analysing: {variable}", fontsize=16, y=0.98)

    # Add table on the top
    ax_table = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_table.axis("off")
    table = ax_table.table(cellText=desc_stats.values,
                           colLabels=desc_stats.columns,
                           rowLabels=desc_stats.index,
                           cellLoc="center",
                           loc="center",
                           bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(desc_stats.columns))))

    # Add padding by scaling the table
    table.scale(1.2, 1.8)  # Adjust these values for horizontal/vertical padding

    # [1,1] Histogram
    ax1 = plt.subplot2grid((3, 2), (1, 0))
    sns.histplot(data[variable], kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Histogram", fontsize=12)
    ax1.set_xlabel(variable)

    # [1,2] Violin Chart
    ax2 = plt.subplot2grid((3, 2), (1, 1), sharex=ax1)
    sns.violinplot(x=data[variable], ax=ax2, color="lightgreen")
    ax2.set_title("Violin Chart", fontsize=12)
    ax2.set_xlabel(variable)

    # [2,1] Box plot
    ax3 = plt.subplot2grid((3, 2), (2, 0), sharex=ax1)
    sns.boxplot(x=data[variable], ax=ax3, color="orange")
    ax3.set_title("Box plot", fontsize=12)
    ax3.set_xlabel(variable)

    # [2,2] Box plot with points overlaid
    ax4 = plt.subplot2grid((3, 2), (2, 1), sharex=ax1)
    sns.boxplot(x=data[variable], ax=ax4, color="lightcoral")
    sns.stripplot(x=data[variable], ax=ax4, color="black", alpha=0.5, jitter=True)
    ax4.set_title("Box plot with points", fontsize=12)
    ax4.set_xlabel(variable)

    # Final Adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def univariate_categorical_variable(data, variable):
    """
    Analyse categorical variable.

    1. Returns transposed describe function in a table format.
    2. Returns a table with level frequency (including percentage and total).
    3. Plot a bar chart with frequency and show values on top.

    Parameters:
        data (pd.DataFrame): Database containing the variable.
        variable (str): Name of the variable to be analysed.

    Returns:
        None

    Usage example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Caregory': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'A', 'B']})
        >>> univariate_categorical_variable(df, 'Caregory')
    """
    # Verify if variable is on the dataframe
    if variable not in data.columns:
        raise ValueError(f"Variable '{variable}' is not in the DataFrame.")

    # 1. Transposed and formated describe
    describe_table = data[variable].describe().to_frame()
    describe_table = describe_table.T
    describe_table.index = [variable]

    # Show formated table
    print("Categorical variable describe:")
    display(describe_table)

    # 2. Frequency of each level (percentage and total)
    frequency_table = data[variable].value_counts().reset_index()
    frequency_table.columns = [variable, 'Frequency']
    frequency_table['Percentage (%)'] = (frequency_table['Frequency'] / len(data) * 100).round(2)

    # Add line for total
    total_row = pd.DataFrame({
        variable: ['Total'],
        'Frequency': [frequency_table['Frequency'].sum()],
        'Percentage (%)': [100.0]
    })
    frequency_table = pd.concat([frequency_table, total_row], ignore_index=True)

    # Show formated table
    print("Frequency table of categorical variable (with percentage and total):")
    display(frequency_table)

    # 3. Frequency bar chart
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=variable, y='Frequency', data=frequency_table[:-1], errorbar=None)

    # Add lables on top of the bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')

    # Configure chart
    plt.title(f'Frequency chart: {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def numeric_variable_analysis_percentile(data, x, y, q=10, chart='none'):
    """
    Sort variable x, divide in percentile and summarize.

    Paameters:
        data (pd.DataFrame): Database containing the variable.
        x (str): Name of the independent varaible.
        y (str): Name of dependent variable.
        q (int): Number of percentile (default: 10).
        chart (str): Chart options: 'p', 'logit', 'both', 'none' (default: 'none').

    Returns:
        pd.DataFrame: DataFrame with summarize statistics by percentile, incluindo:
                      - Percentile
                      - n (number of rows)
                      - Min x
                      - Max x
                      - p (mean y)
                      - logit p

    Usage example:
        >> data = pd.DataFrame({'x': np.random.uniform(0, 100, 1000), 
        'y': np.random.randint(0, 2, 1000)})
        >> result = numeric_variable_analysis_percentile(data, 'x', 'y', q=10, chart='both')
        >> print(result)
    """
    # Certify that y varaible is in a numeric format
    data[y] = pd.to_numeric(data[y], errors='coerce')

    # Sort dataframe by x variable
    data = data.sort_values(by=x).reset_index(drop=True)

    # Create percentiles
    data['percentile'] = pd.qcut(data[x], q=q, labels=[str(i) for i in range(1, q + 1)])

    # Summaraize statistics per percentile
    summary = data.groupby('percentile').agg(
        n=(x, 'count'),
        min_x=(x, 'min'),
        max_x=(x, 'max'),
        p=(y, 'mean')
    ).reset_index()

    # Calculate logit p
    summary['logit_p'] = np.log(summary['p'] / (1 - summary['p']))

    # Adjust to deal where p is 0 or 1
    epsilon = 1e-10  # smal value to adjust 0 e 1
    summary['logit_p'] = np.log(np.clip(summary['p'], epsilon, 1 - epsilon) / 
                                 (1 - np.clip(summary['p'], epsilon, 1 - epsilon)))


    # Chart option
    if chart in ['p', 'logit', 'both']:
        plt.figure(figsize=(12, 6))

        if chart == 'p':
            plt.scatter(summary['percentile'], summary['p'], color='blue')
            plt.title('Percentile chart x p')
            plt.xlabel('Percentile')
            plt.ylabel('p (average of y)')
            plt.grid(True)
            plt.show()

        elif chart == 'logit':
            plt.scatter(summary['percentile'], summary['logit_p'], color='red')
            plt.title('Percentile chart x logit p')
            plt.xlabel('Percentile')
            plt.ylabel('logit p')
            plt.grid(True)
            plt.show()

        elif chart == 'both':
            # Chart side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

            # Percentile Chart x p
            axes[0].scatter(summary['percentile'], summary['p'], color='blue')
            axes[0].set_title('Percentile x p')
            axes[0].set_xlabel('Percentile')
            axes[0].set_ylabel('p (average of y)')
            axes[0].grid(True)

            # Percentile Chart x logit p
            axes[1].scatter(summary['percentile'], summary['logit_p'], color='red')
            axes[1].set_title('Percentile x logit p')
            axes[1].set_xlabel('Percentile')
            axes[1].set_ylabel('logit p')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

    return summary


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

# Explore the dataset structure
print("Dataset Shape:", df.shape)

# %%
df.sample(10)

# %%
df.head()

# %%
df.info()

# %%
df.describe()


# %%
# Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------

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
# Variable: Vendas
# Convert from string to float
df["Sales"] = df["Vendas"].str.replace(",", "").astype(float)

# Count total missing/invalid
missing_count = df["Sales"].isna().sum()
print(f"Total missing or errored rows: {missing_count}")

# Filter and display rows where the conversion failed
error_rows = df[df["Sales"].isna()]
print(error_rows[["Vendas", "Sales"]])

# Check the earliest and latest
print(df["Sales"].describe())


# %%
# Variable: Empresa
df["Company"] = "Comp_" + df["Empresa"].astype(str)

print(df["Company"].unique())

# %%
## Choosing Department
# ------------------------------------------
choose_dept = 'Depto 4'

df_dep = df[df['Departamento'] == choose_dept].copy()

# %%
## Univariate Analysis
# ------------------------------------------
univariate_numeric_variable(df_dep, "Sales")

# %%
univariate_categorical_variable(df_dep, "Company")

# %%
univariate_categorical_variable(df_dep, "Seção")

# %%
univariate_numeric_variable(df_dep, "Vale")

# %%
univariate_numeric_variable(df_dep, "Feriado")

# %%
univariate_numeric_variable(df_dep, "Pagamento")

# %%
# Time series plot of the target variable
plt.figure(figsize=(14, 6))
plt.plot(df_dep['Date'], df_dep['Sales'])
plt.title(f'Daily Sales - {choose_dept}', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()

# %%
## Aggregating by day
# ------------------------------------------
# aggregate by date
df_daily = df_dep.groupby("Date").agg({
    'Sales': 'sum',           # Sum sales for the day
    'Feriado': 'max',         # Max flag (1 if any record is holiday, else 0)
    'Pagamento': 'max',
    'Vale': 'max'
}).reset_index()

# Display result
print(df_daily.head())

# %%
# Time series plot of the target variable
plt.figure(figsize=(14, 6))
plt.plot(df_daily['Date'], df_daily['Sales'])
plt.title(f'Daily Sales - {choose_dept}', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Check if you have all dates in your range
print(f"Date range: {df_daily['Date'].min()} to {df_daily['Date'].max()}")
print(f"Total days in range: {(df_daily['Date'].max() - df_daily['Date'].min()).days + 1}")
print(f"Actual rows: {len(df_daily)}")
print(f"Missing days: {((df_daily['Date'].max() - df_daily['Date'].min()).days + 1) - len(df_daily)}")

# create complete date range
date_range = pd.date_range(
    start=df_daily['Date'].min(), 
    end=df_daily['Date'].max(),
    freq='D'  # Daily frequency
)

# reindex to include all dates
df_complete = df_daily.set_index('Date').reindex(date_range).reset_index()
df_complete.rename(columns={'index': 'Date'}, inplace=True)

# fill missing values
df_complete['Sales'] = df_complete['Sales'].fillna(0.000001)  # Or use other methods
df_complete['Feriado'] = df_complete['Feriado'].fillna(0)     # Default to not holiday
df_complete['Pagamento'] = df_complete['Pagamento'].fillna(0)
df_complete['Vale'] = df_complete['Vale'].fillna(0)

print(df_complete.head())
print(f"Total rows after filling: {len(df_complete)}")

# %%
# Time series plot of the target variable
plt.figure(figsize=(14, 6))
plt.plot(df_complete['Date'], df_complete['Sales'])
plt.title(f'Daily Sales - {choose_dept}', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()

# %%
