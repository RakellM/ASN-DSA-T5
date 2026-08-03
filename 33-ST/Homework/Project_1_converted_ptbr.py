# %% [markdown]
# # **Raquel Marques - Projeto de Séries Temporais**
# 
# **Projeto de Previsão de Vendas: Previsão de Demanda Diária de Departamento Varejista**
# 
# Última Atualização: 2026-08-02
# 
# ---

# %% [markdown]
# Legenda:
# * <span style="color:green">Explicação</span>: Fornece raciocínio detalhado ou contexto para conceitos e processos.
# * <span style="color:purple">Dicas</span>: Oferece conselhos práticos ou melhores práticas para melhorar a eficiência ou os resultados.
# * <span style="color:red">Prática</span>: Destaca passos acionáveis ou exercícios para aplicar os conceitos.
# * <span style="color:blue">Contexto de Negócio</span>: Conecta o trabalho técnico a objetivos ou cenários de negócio relevantes.

# %% [markdown]
# # Sumário
# 
# **Parte 0: Setup**
# - Visão Geral do Projeto e Contexto de Negócio
# - Configuração e Carregamento de Dados
# - Funções Auxiliares
# 
# **Parte 1: Análise Exploratória de Dados e Preparação dos Dados**
# - Dicionário de Dados e Limpeza
# - Escolha do Departamento e Agregação ao Nível Diário
# - Regras de Negócio derivadas para períodos futuros (`isClosed`, `Pagamento`, `Vale`)
# 
# **Parte 2: Análise de Séries Temporais**
# - Passo A: Análise Visual da Série
# - Passo B: Divisão Treino / Holdout
# - Passo C: Verificação de Ruído Branco
# - Passo D: Verificação de Estacionariedade
# - Passo E: Identificação de Componentes (tendência, sazonalidade, ciclo)
# - Passo F: Ajuste de Modelos
#   - F1: Suavização Exponencial
#   - F2: ARIMA / SARIMA (identificação, estimação, variáveis exógenas, coeficientes)
# - Passo G: Seleção do Melhor Modelo
# - Passo H: Diagnóstico de Resíduos
# - Passo I: Métricas de Erro no Holdout
# - Passo J: Previsão para os Próximos Períodos
# 
# **Parte 3: Conclusões**
# - Recomendações de Negócio e Próximos Passos
# - Conclusão
# 
# ---

# %% [markdown]
# # 🔹**PARTE 0**: Setup
# 
# ---
# 
# ## 🔸Visão Geral do Projeto e Contexto de Negócio
# 
# ### Problema de Negócio
# 
# No setor varejista competitivo, a previsão precisa de vendas diárias é crucial para otimizar níveis de estoque, dimensionamento de equipe e fluxo de caixa. A organização atualmente enfrenta desafios com rupturas de estoque e excesso de estoque devido a previsões imprecisas de demanda.
# 
# ### Objetivos do Projeto
# 
# - Desenvolver um modelo robusto de previsão de séries temporais para vendas diárias de um departamento varejista específico
# - Identificar os principais padrões (tendência, sazonalidade, ciclos) que impulsionam as vendas
# - Gerar uma previsão de vendas de 30 dias para março de 2021
# - Fornecer insights de negócio acionáveis com base em análise orientada por dados
# 

# %% [markdown]
# ### Descrição dos Dados
# 
# - Fonte: `Vendas_ASN.csv` (da plataforma ASN Jedi)
# - Variável Alvo: `Vendas` (Vendas) - Departamento 2 ou 4
# - Período: Dados diários, Treinamento: até `Dec/2020`, Holdout: `Jan-Feb/2021`
# - Principais Características:
#     - Variáveis de tempo: `Data` (Data), `Dia_da_semana` (Dia da Semana), `Dia` (Dia), `Mes` (Mês), `Ano` (Ano)
#     - Categóricas: `Empresa` (Empresa), `Departamento` (Departamento), `Secao` (Seção)
#     - Exógenas: `Feriado` (Feriado), `Pagamento` (Dia de Pagamento), `Vale` (Dia de Vale)

# %% [markdown]
# ### Impacto no Negócio
# 
# - **Redução de Custos de Estoque**: 15-20% potencial de redução no excesso de estoque
# - **Aumento de Receita**: Minimização de vendas perdidas por ruptura de estoque
# - **Eficiência Operacional**: Melhor escala de mão de obra alinhada aos padrões de demanda

# %% [markdown]
# ## 🔸Configuração e Carregamento de Dados
# 
# ### Importação de Bibliotecas
# 

# %%
# BIBLIOTECAS
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importações específicas de séries temporais
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates

from scipy.stats import mannwhitneyu, kruskal

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox

import statsmodels.api as sm
import itertools
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import shapiro, jarque_bera


# %%
## CAMINHO
main_dir = os.path.join(os.path.expanduser("~"), 
                           "OneDrive", 
                           "Project_Code")

project_dir = os.path.join(main_dir,
                           "ASN-DSA-T5", 
                           "33-ST",
                           "Homework")

# %% [markdown]
# ### Carregar Dataset 

# %%
# Dataset de feriados
df_holiday = pd.read_csv(os.path.join(project_dir, "data", "holidays.csv"))

# %%
df = pd.read_csv(os.path.join(project_dir, "data", "Vendas ASN_Dados_Finais.csv"))

# %%
# Explorar a estrutura do dataset
print("Formato do Dataset:", df.shape)
df.head()
df.info()
df.describe()

# %% [markdown]
# ## 🔸Funções Auxiliares
# 
# - `univariate_numeric_variable(data, variable)` 
# - `univariate_categorical_variable(data, variable)` 
# - `numeric_variable_analysis_percentile` 
# 

# %%
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
def evaluate_forecast(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# %% [markdown]
# 
# ---
# 

# %% [markdown]
# # 🔹**PARTE 1**: Análise Exploratória de Dados (EDA) e Preparação dos Dados
# 
# ---
# 
# ## EDA 1 - Análise Univariada
# 
# 

# %%
# verificar valores faltantes
df.isnull().sum()

# Listar categorias únicas
print(df["Departamento"].unique())
print(df["Seção"].unique())
print(df["Empresa"].unique())
# print(df["Data_new"].unique())

# %% [markdown]
# ## 🔸<span style="color:green">Dicionário de Dados</span>
# 
# 
# | #   | Variable             | Description                        | Type                    | Raw | Notas                                            |
# | --- | -------------------- | ---------------------------------- | ----------------------- | --- | ------------------------------------------------ |
# | 1   | Data (dia, mês, ano) | Data no formato D/M/AA          | Date                    | Y   |                                                  |
# | 2   | Data_new             | Data no formato D/M/AAYY        | Date                    | Y   |                                                  |
# | 3   | Dia_da_semana        | Dia da Semana                        | Quantitativa Discreta   | Y   | Números de 1 to 7                              |
# | 4   | Dia                  | Day                                | Quantitativa Discreta   | Y   | Números de 1 to 31                             |
# | 5   | Mes                  | Month                              | Quantitativa Discreta   | Y   | Números de 1 to 12                             |
# | 6   | Ano                  | Year                               | Quantitativa Discreta   | Y   | Números de 2018 to 2021                        |
# | 7   | Empresa              | Company                            | Qualitativa Nominal     | Y   | Números de 1 to 23 (but not all of them)       |
# | 8   | Feriado              | Indicador de feriado                    | Qualitativa Binária      | Y   |                                                  |
# | 9   | Pagamento            | Indicador de dia de pagamento               | Qualitativa Binária      | Y   |                                                  |
# | 10  | Vale                 | Indicador de dia de pagamento do vale-alimentação       | Qualitativa Binária      | Y   |                                                  |
# | 11  | Vendas               | vendas (ALVO)                     | Quantitativa Contínua | Y   |                                                  |
# | 12  | Departamento         | Departamento                         | Qualitativa Nominal     | Y   | `Depto 1` Departamento number varying form 1  to 7 |
# | 13  | Seção                | Seção                            | Qualitativa Nominal     | Y   | `Seção 27` Seção number varying from 1 to 29   |
# | 14  | Date                 | Data no formato AAAA-MM-DD      | Date                    | N   |                                                  |
# | 15  | Sales                | Esta é a variável de Vendas (alvo) | Quantitativa Contínua | N   |                                                  |
# 
# 
# 
# 

# %% [markdown]
# ### ▪️<span style="color:purple">Notas</span>
# 
# - `Data_new` é o campo de data.
#     - Verificar the first and last date, identify is there are missing days and how it can be inputed.
#         - Remember if the level of previsãoing is daily, we need ocmplete years starting from Jan-01 all the way to Dec-31.
# 
# - `Data` com ano de apenas 2 caracteres: como os dados começam em 2018 não é problema, mas poderia ser se houvesse datas antes de 2000 ou depois de 2100.
#     - **Decisão**: `Data_new` é a data das vendas com 4 dígitos para o ano, então podemos remover `Date`.
# 
# - `Empresa` é numérica no banco de dados, mas é uma variável categórica por natureza, pois representa empresas diferentes.
#     - Não tem 23 empresas como os números indicariam — algumas estão ausentes, indício de que se trata mais de uma categórica.
#     - **Decisão**: Se for usada, precisa ser tratada como categórica.
# 
# - Variáveis de flag: `Feriado`, `Vale`, `Pagamento`
#     - **Verificar**: se são verdadeiramente binárias, se há missing e como traduzir para dias faltantes e futuros.
# 
# - `Vendas` é a nossa variável de vendas.
#     - Variável ALVO
#     - **Verificar**:
#         - Temos `sales = 0`? Significa que não houve vendas, mas a empresa abriu?
#         - Datas faltantes significam 0 vendas? Ou que a empresa não funciona nesse dia?
#         - Vendas negativas podem ocorrer? Se não, como tratar esse problema?
#         
# 

# %% [markdown]
# ## 🔸Limpeza e Pré-processamento dos Dados

# %% [markdown]
# ### <span style="color:green">Variável: Date</span>
# 
# - Observe que a data começa em `2018-01-02`, então já estamos sem o dia 01 de janeiro no dataset.
# 

# %%
# Variable : Data_new
# Converter data de string para data
df["Date"] = pd.to_datetime(df["Data_new"], errors='coerce')

# Contar total de datas faltantes/inválidas
missing_count = df["Date"].isna().sum()
print(f"Total de linhas faltantes ou com erro: {missing_count}")

# Filtrar e exibir linhas onde a conversão falhou
error_rows = df[df["Date"].isna()]
print(error_rows[["Data_new", "Date"]])

# Verificar as datas mais antiga e mais recente para detectar erros de digitação
print(df["Date"].describe())

# %% [markdown]
# ### <span style="color:green">Variável Vendas</span>
# 
# - Observe que temos vendas negativas que precisam ser tratadas.
# - Além disso, parece haver muitos `sales = 0`. Dependendo do nível dos dados com que trabalharemos, isso pode ser um problema.

# %%
# Variable: Vendas
# Converter de string para float
df["Sales"] = df["Vendas"].str.replace(",", "").astype(float)

# Contar total de faltantes/inválidos
missing_count = df["Sales"].isna().sum()
print(f"Total de linhas faltantes ou com erro: {missing_count}")

# Filtrar e exibir linhas onde a conversão falhou
error_rows = df[df["Sales"].isna()]
print(error_rows[["Vendas", "Sales"]])

# Verificar o mais antigo e o mais recente
print(df["Sales"].describe())

univariate_numeric_variable(df, "Sales")

# %% [markdown]
# #### <span style="color:green">Variável Vendas_adj</span>

# %%
df["Sales_adj"] = df["Sales"].clip(lower=0)

# Verificar o mais antigo e o mais recente
print(df["Sales_adj"].describe())

univariate_numeric_variable(df, "Sales_adj")

# %% [markdown]
# ### <span style="color:green">Variável: Company</span>
# 
# - Decidimos criar uma string para cada empresa.
# - Observe que `Comp_1` é a que mais aparece no banco e `Comp_23` a que menos aparece.

# %%
# Variable: Empresa
df["Company"] = "Comp_" + df["Empresa"].astype(str)

print(df["Company"].unique())

univariate_categorical_variable(df, "Company")

# %% [markdown]
# ### <span style="color:green">Variável: Seção</span>
# 
# - `Seção 23` é a mais frequente no banco (representando 5,8%).
# - `Seção 29` é a menos presente no banco, com apenas 14 registros.

# %%
univariate_categorical_variable(df, "Seção")

# %% [markdown]
# ### <span style="color:green">Variável: Departamento</span>
# 
# - `Depto 3` é o mais frequente, representando 38% dos dados.
# - `Depto 4` e `Depto 2` representam ~25% dos dados
# - Todos os outros departamentos estão abaixo de 10%.
# 

# %%
univariate_categorical_variable(df, "Departamento")

# %% [markdown]
# ### ▫️<span style="color:purple">Seção Final Notas</span>
# 
# Analisamos cada variável separadamente, mas cada nível de combinação pode nos dar outra direção de como os dados podem ser usados para obter a melhor previsão.
# 
# - Company - Departamento - Seção : 
#     - Usando essa combinação, vemos que nem todas as empresas têm o mesmo departamento nem as mesmas seções ou número de seções. 
#     - O número de dias também não é o mesmo.
# 
# 
# - Departamento - Seção : 
#     - Usando essa combinação, vemos que nem todos os departamentos têm as mesmas seções ou número de seções.
#     - O número de dias também não é o mesmo.
# 
# Since the initial problem told us to focus on Departamento 2 or 4, let's group our data athe Departamento level, and _ignore_ for now Company and Seção variables as they seem to split the information too much and can cause noise when previsãoing.

# %%
df.groupby(['Company', 'Departamento', 'Seção']).size()

# %%
df.groupby(['Departamento', 'Seção']).size()

# %% [markdown]
# ## 🔸Escolha do Departamento e Agregação ao Nível Diário

# %%
# agregar por Departamento & Data
df_depts = df.groupby(["Departamento", "Date"]).agg({
    'Sales_adj': 'sum',           # Somar vendas do dia
    'Feriado': 'max',             # Flag máxima (1 se qualquer registro for feriado, senão 0)
    'Pagamento': 'max',
    'Vale': 'max'
}).reset_index()

# %%
univariate_categorical_variable(df_depts, "Departamento")

# univariate_categorical_variable(df_depts, "Feriado")

# univariate_categorical_variable(df_depts, "Pagamento")

# univariate_categorical_variable(df_depts, "Vale")

# univariate_numeric_variable(df_depts, "Sales_adj")

# %%
# gráfico Departamento & Vendas por Data

plt.figure(figsize=(14, 8))
sns.lineplot(data=df_depts, x='Date', y='Sales_adj', hue='Departamento')
plt.title('Sales by Department Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(nrows=len(df_depts['Departamento'].unique()), 
                         figsize=(12, 3*len(df_depts['Departamento'].unique())), 
                         sharex=True)

for ax, (dept, group) in zip(axes, df_depts.groupby('Departamento')):
    ax.plot(group['Date'], group['Sales_adj'], linestyle='-')
    ax.set_title(f'Department: {dept}')
    ax.set_ylabel('Sales')
    ax.grid(True, alpha=0.3)

plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Escolher Departamento
# 
# A partir deste momento, we have aggregated Departamento by dates, and we will now choose a department to focus on.
# 
# There is no more Company or Seção.

# %%
# escolhendo departamento
choose_dept = 'Depto 4'
print(f"Trabalharemos com: {choose_dept}")

df_dep = df_depts[df_depts['Departamento'] == choose_dept].copy()

# %% [markdown]
# ### Preencher datas faltantes
# 
# Precisamos criar um dataset com todas as datas do período que vamos analisar.
# 
# 
#     

# %%
# obter ano mínimo
min_year = df_dep['Date'].dt.year.min()

# criar data de início como 1º de janeiro daquele ano
start_date = pd.Timestamp(f'{min_year}-01-01')

# obter a data final
end_date = df_dep['Date'].max() 

# criar intervalo completo de datas
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# obter departamentos únicos
departments = df_dep['Departamento'].unique()

# criar todas as combinações de datas e departamentos
df_complete = pd.DataFrame(
    [(date, dept) for date in all_dates for dept in departments],
    columns=['Date', 'Departamento']
)

# mesclar com dados originais
df_complete = df_complete.merge(df_dep, on=['Date', 'Departamento'], how='left')

# Criar colunas diferentes para datas, mês, ano e combinação
df_complete['year'] = df_complete['Date'].dt.year
df_complete['month'] = df_complete['Date'].dt.month
df_complete['year_month'] = df_complete['Date'].dt.to_period('M')
df_complete['weekday'] = df_complete['Date'].dt.day_name()



# %%
# df_complete

# %%
# Análise de datas faltantes
df_complete[df_complete["Feriado"].isna()]

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# Todas as datas faltantes eram Ano Novo e Natal!
# 
# - **Question**
#     - _Isso pode significar que nunca houve vendas nessas datas, possivelmente porque não abrem?_
#     - _Podemos criar uma flag para indicar que nunca haverá vendas nesse dia no futuro?_

# %% [markdown]
# #### Variável: isClosed
# 
# Vamos criar uma flag para as datas faltantes no dataset original, porque essas datas sempre estarão ausentes no futuro.
# 
# - Ocorre em todo 25 de dezembro e 1º de janeiro

# %%
df_complete['isClosed'] = ((df_complete['Date'].dt.month == 12) & (df_complete['Date'].dt.day == 25)) | \
                          ((df_complete['Date'].dt.month == 1) & (df_complete['Date'].dt.day == 1))

df_complete['isClosed'] = df_complete['isClosed'].astype(int) 

# %%
# Assign 0 to the other variables

df_complete['Feriado'] = df_complete['Feriado'].fillna(0)    
df_complete['Pagamento'] = df_complete['Pagamento'].fillna(0)
df_complete['Vale'] = df_complete['Vale'].fillna(0)
df_complete['Sales_adj'] = df_complete['Sales_adj'].fillna(0)

# %% [markdown]
# ### Variável: Feriado

# %% [markdown]
# #### <span style="color:purple">Dataset Externo:</span> `holidays.csv`
# 
# Para permitir o mapeamento preciso das flags de feriado em datas futuras, obtivemos dados de feriados brasileiros da [Feriados API](https://feriadosapi.com/docs#introducao). 
# Esse dataset externo fornece uma referência padronizada de feriados em diferentes períodos e nos permite avaliar sistematicamente como os feriados estão representados na nossa série temporal.
# 

# %% [markdown]
# Dictionary
# 
# 
# | #   | Variable    | Description                                                                                                                  | Type                  | Raw | Notas |
# | --- | ----------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------- | --- | ----- |
# | 1   | id          | unique ID of holiday date                                                                                                    | Qualitativa Nominal   | Y   |       |
# | 2   | data        | Holiday date, format `DD/MM/YYYY`                                                                                            | Date                  | Y   |       |
# | 3   | nome        | Holiday name                                                                                                                 | Qualitativa Nominal   | Y   |       |
# | 4   | tipo        | Which holiday databse the holiday was extracted: National / State / State Capital<br>`NACIONAL` /  `ESTADUAL` /  `MUNICIPAL` | Qualitativa Nominal   | Y   |       |
# | 5   | descricao   | Holiday descriptive long                                                                                                     | Qualitativa Nominal   | Y   |       |
# | 6   | uf          | Brazilian State. <br>Only present in **tipo** = `ESTADUAL` /  `MUNICIPAL`                                                    | Qualitativa Nominal   | Y   |       |
# | 7   | codigo_ibge | City code according to IBGE.<br>Only present in **tipo** =  `MUNICIPAL`<br>                                                  | Qualitativa Nominal   | Y   |       |
# | 8   | bancario    | Flag for a bank holiday                                                                                                      | Qualitativa Binária    | Y   |       |
# | 9   | year        | Holiday date year extracted.                                                                                                 | Quantitativa Discreta | Y   |       |
# | 10  | type        | same as `tipo`<br>`national` /  `state` /  `capital`                                                                         | Qualitativa Nominal   | Y   |       |
# | 11  | state       | same as `uf`                                                                                                                 | Qualitativa Nominal   | Y   |       |
# | 12  | ibge_code   | same as `codigo_ibge`                                                                                                        | Qualitativa Nominal   | Y   |       |
# | 13  | Date        | Convert from a regular english date format `YYYY-MM-DD`                                                                      | Date                  | N   |       |
# 
# 
# 

# %%
df_holiday["Date"] = pd.to_datetime(df_holiday["data"], format='%d/%m/%Y', errors='coerce')
# df_holiday.head()
# df_holiday[df_holiday["Date"] == '2019-10-12']

df_holiday_simple = df_holiday[["Date", "nome"]].copy()
df_holiday_unique = df_holiday_simple.drop_duplicates(subset=['Date'], keep='first')

# df_holiday_simpleSP = df_holiday_simple[df_holiday_simple["state"] == 'SP'].copy()
# df_holiday_simpleSP.head()

# Check if there are any duplicate dates
has_duplicates = df_holiday_unique['Date'].duplicated().any()
print(f"Are there duplicate dates? {has_duplicates}")

# See which dates are duplicated
duplicate_dates = df_holiday_unique[df_holiday_unique['Date'].duplicated(keep=False)]
print("Duplicate dates:")
print(duplicate_dates['Date'].value_counts())



# %%
df_complete1 = df_complete.merge(df_holiday_unique, on=['Date'], how='left')

# %%
df_map_holidays = df_complete1[df_complete1["Feriado"] == 1]

print(f"Total holidays in the database: {df_map_holidays.shape}")
print(f"Holidays missing description: {df_map_holidays[df_map_holidays["nome"].isna()].shape}")

# %% [markdown]
# #### Investigando registros com `Feriado = 1` **sem** descrição de feriado
# 
# Registros em que o indicador de feriado está ativo (`Feriado = 1`), mas nenhum nome de feriado é fornecido.  
# Examinamos a frequência desses casos, sua distribuição ao longo do tempo e o potencial impacto na análise de séries temporais.

# %%
df_map_holidays[df_map_holidays["nome"].isna()]

# %% [markdown]
# #### Investigando registros com `Feriado = 1` **com** descrição de feriado
# 
# Registros em que o indicador de feriado está ativo (`Feriado = 1`) e um nome de feriado é fornecido.  
# Examinamos os feriados únicos, suas contagens de ocorrência e padrões temporais para entender melhor seu potencial impacto na série.

# %%
# what are the holidays being correctly flag?
df_map_holidays.groupby(['nome']).size()

# %% [markdown]
# #### Investigando registros com `Feriado = 0` **com** descrição de feriado
# 
# Esses registros têm a flag de feriado definida como 0, mas um nome/descrição de feriado está presente.  
# This may reflect the granularity of the holiday dataset, for example, local or small-city holidays that do not significativoly affect main businesses and are therefore not flagged as relevant.  
# Esta seção verifica a frequência desses casos, quais feriados aparecem e se devem ser tratados como problemas de qualidade de dados ou comportamento esperado.

# %%
# what are the non holidays that has description?
df_map_NOTholidays = df_complete1[df_complete1["Feriado"] == 0]
df_map_NOTholidays.groupby(['nome']).size()

# %% [markdown]
# #### Flags de feriado inconsistentes: mesma data marcada como feriado e não-feriado
# 
# Investigamos casos em que o mesmo dia do calendário aparece com `Feriado = 1` em alguns anos e `Feriado = 0` em outros.  
# Essa análise ajuda a detectar possíveis inconsistências no dataset de feriados ou diferenças na forma como os feriados foram registrados ao longo do tempo.

# %%
# Check if any holiday names have inconsistent flags
inconsistent_holidays = df_complete1.groupby('nome')['Feriado'].nunique()
inconsistent_holidays = inconsistent_holidays[inconsistent_holidays > 1].index.tolist()

print(f"Holidays with inconsistent flags: {len(inconsistent_holidays)}")
print(inconsistent_holidays)

# View the actual inconsistencies
df_inconsistent = df_complete1[df_complete1['nome'].isin(inconsistent_holidays)]
df_inconsistent[['Date', 'nome', 'Feriado']].sort_values(['nome', 'Date'])

# %%
# df_complete1[df_complete1['nome'] == "Proc. República Rio Grandense"]

# %%
# df_holiday[df_holiday['Date'] == '2019-09-20']

# %% [markdown]
# #### 📌 <span style="color:red">REGRA</span>: Marcação de Feriados Futuros
# 
# **Principais Aprendizados**:
# - As descrições de feriados são consistentes e mapeáveis.
#     - Todos os feriados no dataset podem ser vinculados a um nome descritivo (campo `nome`). 
#     - Isso torna viável usar essas descrições como referência confiável para sinalizar feriados em dados futuros.
# 
# - Descrições de não-feriados exigem cautela.
#     - Algumas descrições aparecem no dataset, mas não foram sinalizadas como feriados (`Feriado = 0`). 
#     - Isso pode ser devido a variações regionais — certos feriados são observados apenas em cidades ou estados específicos, e não nacionalmente. 
#     - Esses casos devem ser tratados com cuidado ao aplicar a flag a períodos futuros.
# 
# - Sinalização inconsistente de feriados observada para certos eventos.
#     - O feriado _"Proc. República Rio Grandense"_ foi sinalizado como feriado (`Feriado = 1`) em 2019, mas não em 2018 ou 2020. 
#     - Essa inconsistência levanta questões sobre se esse evento deve ser considerado feriado em previsões futuras.
#     - **Decisão**: Por enquanto, assumiremos esse feriado como NÃO feriado (`Feriado = 0`) ao mapear datas futuras, a menos que contexto ou dados adicionais indiquem o contrário.

# %%
holiday_descriptions = df_map_holidays.groupby(['nome']).size()

holiday_descriptions


# %% [markdown]
# ##### Feriados a serem usados para a Flag de Feriado

# %%
# Get the list of holiday names from holiday_descriptions
holiday_names = holiday_descriptions.index.tolist()

# Filter the main dataframe
df_holiday_filtered = df_holiday_unique[df_holiday_unique['nome'].isin(holiday_names)]

df_holiday_filtered

# %% [markdown]
# ### Variável: Pagamento

# %%
# Add .reset_index(name='count') at the end of line 1
df_pgto = df_complete[df_complete['Pagamento'] == 1].copy()
check_payday = df_pgto.groupby(['Date']).size().reset_index(name='count')

# Now 'Date' is a normal column, so this will work perfectly
check_payday["Date"] = pd.to_datetime(check_payday["Date"])

# Extract your weekday columns
check_payday["day_name"] = check_payday["Date"].dt.day_name()
check_payday["day_num"] = check_payday["Date"].dt.weekday

check_payday

# %% [markdown]
# #### Dia do Mês
# 
# Verificaring what is the most frequent **day of the month** payment day is likely to fall.
# 
# - Maior frequência no dia 6 e 7
# - Varia entre 4 e 8, significando que tende a cair na primeira semana do mês
# - **Question**:
#     - _O dia da semana seria mais consistente?_

# %%
# Extract the day number of the month (e.g., 5, 6, 7)
check_payday["day_of_month"] = check_payday["Date"].dt.day

# Count how many times each day of the month appears
payday_frequencies = check_payday.groupby("day_of_month").size().reset_index(name="payment_count")

# Sort by the highest frequency to see the dominant day
payday_frequencies = payday_frequencies.sort_values(by="payment_count", ascending=False)

print("\nPayment frequency per day of month:")
print(payday_frequencies)


# %% [markdown]
# #### Dia da Semana
# 
# Verificaring now for **day of the week**.
# 
# - Sexta-feira tem maior frequência
# - O pagamento **nunca** ocorre no fim de semana
# - **Question**:
#     - _Alguma razão para não ser apenas na sexta-feira?_
# 

# %%
# Count how many times each day appears
payday_frequencies2 = check_payday.groupby("day_name").size().reset_index(name="payment_count")

# Sort by the highest frequency to see the dominant day
payday_frequencies2 = payday_frequencies2.sort_values(by="payment_count", ascending=False)

print("\nPayment frequency per week day:")
print(payday_frequencies2)

# %% [markdown]
# #### Day of Month, Dia da Semana, Holiday, Payment and isClosed
# 
# - O pagamento no dia 6 é mais frequente do que ser na sexta-feira. 
# - **Questions** 
#     - _Se o dia 6 cair no fim de semana, o pagamento foi movido para o dia 7?_
#         - Apenas quando o dia 6 cai em um DOMINGO. 
#             - Se o dia 6 for sábado → Pagamento move para sexta (dia 5)
#             - Se o dia 6 for domingo → Pagamento move para segunda (dia 7)
# 
#     - _Se o dia 6 for feriado, o pagamento foi movido para o dia 7?_
#         - Não. Feriados tipicamente movem o pagamento para o dia útil ANTERIOR.
#             - Se o dia 6 for feriado de sexta → Pagamento move para quinta (dia 4)
#             - Se o dia 6 for domingo e o dia 7 (segunda) também for feriado → Pagamento move para terça (dia 8)
# 
#     - _Quantos dias úteis há do início do mês até o dia de pagamento, sem contar feriados e dias fechados (sem vendas)?_
#         - O pagamento tipicamente ocorre no 5º dia útil do mês.
#         - Distribuição de frequência:
#             - 3 dias úteis: 28 meses (70%) ← MAIS COMUM
#             - 4 dias úteis: 12 meses (30%)
# 
# 
# 

# %%
# Get payment days only
payments = df_complete[df_complete['Pagamento'] == 1].copy()
payments['Date'] = pd.to_datetime(payments['Date'])
payments['day_of_month'] = payments['Date'].dt.day
payments['month'] = payments['Date'].dt.month
payments['year'] = payments['Date'].dt.year

# For each payment, get the first business day of that month
# First, get all business days (not weekend, not Feriado, not isClosed)
df_complete['is_business_day'] = (
    (df_complete['Date'].dt.weekday < 5) &  # Monday-Friday
    (df_complete['Feriado'] != 1) &
    (df_complete['isClosed'] != 1)
)

# For each month, find the first business day
first_biz_days = df_complete[df_complete['is_business_day']].groupby('year_month')['Date'].min().reset_index()
first_biz_days.columns = ['year_month', 'first_business_day']

# Merge with payments
payments = payments.merge(first_biz_days, on='year_month', how='left')

# For each payment, count Feriado and isClosed in the period
def count_flags(row):
    mask = (df_complete['Date'] >= row['first_business_day']) & (df_complete['Date'] <= row['Date'])
    period_data = df_complete[mask]
    return pd.Series({
        'feriado_count': period_data['Feriado'].sum(),
        'closed_count': period_data['isClosed'].sum(),
        'weekend_count': (period_data['Date'].dt.weekday >= 5).sum()
    })

# Apply the function to get counts
counts = payments.apply(count_flags, axis=1)
payments = pd.concat([payments, counts], axis=1)

# Calculate days from first business day
payments['days_from_first_biz'] = (payments['Date'] - payments['first_business_day']).dt.days - payments['weekend_count']
payments['biz_days_from_first_biz'] = payments['days_from_first_biz'] - payments['feriado_count'] - payments['closed_count']

# Add total days in the month
payments['total_days_in_month'] = payments['Date'].dt.days_in_month

# Add total business days in the month (excluding weekends, Feriado, isClosed)
def count_total_biz_days(row):
    year = row['year']
    month = row['month']
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = start_date + pd.offsets.MonthEnd(0)
    month_dates = pd.date_range(start=start_date, end=end_date)
    
    biz_count = 0
    for d in month_dates:
        day_data = df_complete[df_complete['Date'] == d]
        if len(day_data) > 0:
            if d.weekday() < 5 and day_data['Feriado'].values[0] != 1 and day_data['isClosed'].values[0] != 1:
                biz_count += 1
    return biz_count

payments['total_biz_days_in_month'] = payments.apply(count_total_biz_days, axis=1)

# Create final result table
result_df = payments[['year_month', 'day_of_month', 'first_business_day', 'weekday', 
                      'feriado_count', 'closed_count', 'weekend_count', 'days_from_first_biz', 
                      'biz_days_from_first_biz', 'total_days_in_month', 'total_biz_days_in_month']].copy()
result_df.columns = ['Month', 'Payment_Day', 'First_Business_Day', 'Payment_Weekday', 
                     'Feriado_Count', 'isClosed_Count', 'Weekend_Count', 'Days_From_First_Biz', 
                     'Biz_Days_From_First_Biz', 'Total_Days_In_Month', 'Total_Biz_Days_In_Month']

print("\nPAYMENT ANALYSIS TABLE")
print("\n" + "="*100)
print(result_df.to_string(index=False))
print("="*100)


# %%
# Frequency of business days from first business day
print("FREQUENCY OF BUSINESS DAYS FROM FIRST BUSINESS DAY")
freq_biz_days = payments.groupby('biz_days_from_first_biz').size().reset_index(name='count')
freq_biz_days.columns = ['Business_Days', 'Number_of_Months']
print(freq_biz_days.sort_values('Number_of_Months', ascending=False).to_string(index=False))

# %%
print("PAYMENT BUSINESS DAYS BY BUSINESS MONTH SIZE")
print(result_df.groupby('Total_Biz_Days_In_Month')['Biz_Days_From_First_Biz'].value_counts().unstack(fill_value=0))

# %%
print("PAYMENT BUSINESS DAYS BY MONTH SIZE")
print(result_df.groupby('Total_Days_In_Month')['Biz_Days_From_First_Biz'].value_counts().unstack(fill_value=0))

# %% [markdown]
# #### 📌<span style="color:red">REGRA</span>: Dias de Pagamento Futuros
# 
# > Pagar no 3º ou 4º dia útil do mês, contando a partir do primeiro dia útil e excluindo fins de semana, feriados e dias fechados.
# 
# 1. Pagar no 3º ou 4º dia útil a partir do primeiro dia útil do mês
# 2. Dias úteis = dias da semana (segunda a sexta) que NÃO são Feriado e NÃO são isClosed
# 3. O 6º dia do calendário é o alvo, mas apenas se cair no 3º ou 4º dia útil
# 4. Se o dia 6 for sábado → Pagar no dia 5 (sexta)
# 5. Se o dia 6 for domingo → Pagar no dia 7 (segunda)
# 6. Se o dia 6 for feriado ou fechado → Pagar no dia útil anterior
# 7. Se o dia ajustado (das regras 4-6) também for feriado ou fechado → Mover para o dia útil anterior
# 8. Se o dia 6 cair em dia útil, mas for apenas o 2º dia útil a partir do primeiro → Mover para o 7º ou 8º para alcançar o 3º ou 4º dia útil
# 
# 

# %%
def get_payment_date(year, month, df_complete):
    """
    Returns the 4th business day of the month
    Business day = not weekend, not Feriado, not isClosed
    """
    # Get all dates in the month
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = start_date + pd.offsets.MonthEnd(0)
    month_dates = pd.date_range(start=start_date, end=end_date)
    
    # Create a dictionary for quick lookup of Feriado and isClosed
    date_dict = {}
    for _, row in df_complete.iterrows():
        date_dict[row['Date']] = {
            'Feriado': row['Feriado'],
            'isClosed': row['isClosed']
        }
    
    # Find business days
    biz_days = []
    for d in month_dates:
        # Skip weekends
        if d.weekday() >= 5:
            continue
        
        # Check if we have data for this day
        if d not in date_dict:
            continue
        
        # Skip if Feriado or isClosed
        if date_dict[d]['Feriado'] == 1 or date_dict[d]['isClosed'] == 1:
            continue
        
        biz_days.append(d)
    
    # Return the 4th business day (index 3)
    if len(biz_days) >= 4:
        return biz_days[3]  # 4th business day
    else:
        print(f"Warning: Only {len(biz_days)} business days found for {year}-{month:02d}")
        return None

# Test for a specific month
print("Testing January 2018:")
test_date = get_payment_date(2018, 1, df_complete)
print(f"Calculated: {test_date}")  # Should be 2018-01-05

print("\nTesting February 2018:")
test_date = get_payment_date(2018, 2, df_complete)
print(f"Calculated: {test_date}")  # Should be 2018-02-06

# Usage
# payment_date = get_payment_date(2021, 1, df_complete)
# print(f"Payment should be on: {payment_date}")

# %%
# Store all payment dates in a DataFrame
results_payday_range = []
for year in range(2018, 2022):
    for month in range(1, 13):
        payment_date = get_payment_date(year, month, df_complete)
        if payment_date:
            results_payday_range.append({
                'year': year,
                'month': month,
                'payment_date': payment_date,
                'day_of_month': payment_date.day,
                'weekday': payment_date.strftime('%A')
            })

results_payday_range_df = pd.DataFrame(results_payday_range)
results_payday_range_df

# %%
df_complete = df_complete.merge(results_payday_range_df[['payment_date']], left_on='Date', right_on='payment_date', how='left')

# Add flags
df_complete['is_calculated_payment_day'] = df_complete['Date'] == df_complete['payment_date']
df_complete['payment_matches_rule'] = (df_complete['Pagamento'] == 1) & (df_complete['is_calculated_payment_day'] == True)

df_complete

# %% [markdown]
# #### ▪️<span style="color:purple">Note</span>
# 
# Temos 2 meses que tiveram 2 dias de pagamento: fev/2018 e dez/2019. A regra criada acertou um dos dias e, como vamos prever apenas um dia de pagamento por mês, consideramos que o modelo estava correto.
# 
# Assim, a regra consegue capturar mais de 75% dos dias de pagamento, então seguiremos com ela para encontrar os dias de pagamento futuros.
# 
# |              | contagem geral | contagens sem duplicatas |
# | ------------ | ------------- | ------------------------- |
# | Corresponde à regra | 27 (67.5%)    | 29 (76%)                  |
# | Não corresponde   | 13 (32.5%)    | 9 (24%)                   |
# | Total        | 40            | 38                        |
# 

# %%
# Filter and count matches
payment_actual = df_complete[df_complete['Pagamento'] == 1]  # Actual payments
payment_calculated = df_complete[df_complete['is_calculated_payment_day'] == True]  # Calculated payment days

# Count matches and mismatches
total_payments = len(payment_actual)
matches = len(payment_actual[payment_actual['is_calculated_payment_day'] == True])
mismatches = total_payments - matches

print(f"Total payments: {total_payments}")
print(f"Matches rule: {matches} ({matches/total_payments*100:.1f}%)")
print(f"Mismatches: {mismatches} ({mismatches/total_payments*100:.1f}%)")

# Show mismatches
print("\nMISMATCHES")
mismatch_df = payment_actual[payment_actual['is_calculated_payment_day'] == False]
print(mismatch_df[['Date', 'Pagamento', 'payment_date']].to_string(index=False))

# Show matches
print("\nMATCHES")
match_df = payment_actual[payment_actual['is_calculated_payment_day'] == True]
print(match_df[['Date', 'Pagamento', 'payment_date']].to_string(index=False))

# %%
# check if any rule was missed when the rule
# print(df_complete[((df_complete['Pagamento'] == 1) | (df_complete['is_calculated_payment_day'] == True)) & (df_complete['payment_matches_rule'] == False)])

# %%
# df_complete[df_complete['year_month'] == '2021-02'].head(10)

# %% [markdown]
# ### Variável: Vale
# 

# %%
df_vale = df_complete[df_complete['Vale'] == 1].copy()
check_vale = df_vale.groupby(['Date']).size().reset_index(name='count')

# Now 'Date' is a normal column, so this will work perfectly
check_payday["Date"] = pd.to_datetime(check_vale["Date"])

# Extract your weekday columns
check_vale["day_name"] = check_vale["Date"].dt.day_name()
check_vale["day_num"] = check_vale["Date"].dt.weekday

check_vale

# %%
# Extract the day number of the month (e.g., 5, 6, 7)
check_vale["day_of_month"] = check_vale["Date"].dt.day

# Count how many times each day of the month appears
vale_frequencies = check_vale.groupby("day_of_month").size().reset_index(name="vale_count")

# Sort by the highest frequency to see the dominant day
vale_frequencies = vale_frequencies.sort_values(by="vale_count", ascending=False)

print("\nVale frequency per day of month:")
print(vale_frequencies)


# %%
# Count how many times each day appears
vale_frequencies2 = check_vale.groupby("day_name").size().reset_index(name="vale_count")

# Sort by the highest frequency to see the dominant day
vale_frequencies2 = vale_frequencies2.sort_values(by="vale_count", ascending=False)

print("\nVale frequency per week day:")
print(vale_frequencies2)

# %%
vales = df_complete[df_complete['Vale'] == 1].copy()
vales['Date'] = pd.to_datetime(vales['Date'])
vales['day_of_month'] = vales['Date'].dt.day
vales['month'] = vales['Date'].dt.month
vales['year'] = vales['Date'].dt.year
vales['weekday'] = vales['Date'].dt.day_name()

def get_biz_days_for_month(year, month, df):
    """Return the ordered list of true business days for a given month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = start + pd.offsets.MonthEnd(0)
    month_dates = pd.date_range(start, end)

    date_dict = {
        row['Date']: row
        for _, row in df.iterrows()
    }

    biz = []
    for d in month_dates:
        if d.weekday() >= 5:                       # weekend
            continue
        if d not in date_dict:                     # missing day
            continue
        if date_dict[d]['Feriado'] == 1 or date_dict[d]['isClosed'] == 1:
            continue
        biz.append(d)
    return biz

# Calculate the true 1-based position of each Vale day
results = []
for _, row in vales.iterrows():
    d = row['Date']
    biz = get_biz_days_for_month(d.year, d.month, df_complete)

    if d in biz:
        pos = biz.index(d) + 1                     # 1-based
        from_end = len(biz) - pos + 1
    else:
        pos = None                                 # weekend or closed day
        from_end = None

    results.append({
        'Date': d,
        'day_of_month': d.day,
        'weekday': d.strftime('%A'),
        'month': d.month,
        'year': d.year,
        'biz_position': pos,                       # ← correct ordinal
        'total_biz_days': len(biz),
        'from_end': from_end
    })

result_df2 = pd.DataFrame(results)

print("=" * 80)
print("VALE ANALYSIS")
print("=" * 80)
print(result_df2.to_string(index=False))

print("\n\nFREQUENCY OF TRUE BUSINESS-DAY POSITION (1-based)")
print(result_df2['biz_position'].value_counts().sort_index())

print("\n\nFREQUENCY FROM THE END OF THE MONTH")
print(result_df2['from_end'].value_counts().sort_index())

# %%
print("\nFREQUENCY OF WEEK DAY")
print(result_df2.groupby('weekday').size().reset_index(name='count').sort_values('count', ascending=False))

print("\nFREQUENCY OF DAY")
print(result_df2.groupby('day_of_month').size().reset_index(name='count').sort_values('count', ascending=False))

# %%
print("\nFREQUENCY OF VALE PER MONTH")
result_df2.groupby('month').size().reset_index(name='count').sort_values('count', ascending=False)

# %%
print("VALE WEEK DAYS BY BUSINESS DAY")
print(result_df2.groupby('weekday')['biz_position'].value_counts().unstack(fill_value=0))

# %%
print("VALE DAY BY BUSINESS DAY")
print(result_df2.groupby('day_of_month')['biz_position'].value_counts().unstack(fill_value=0))

# %% [markdown]
# #### 📌<span style="color:red">REGRA</span>: Dias de Vale Futuros
# 
# > O Vale é pago no **17º dia útil** do mês.  
# > Em novembro e dezembro, um pagamento adicional é feito no **16º dia útil**.
# 
# 1. **Regra Principal**: Pagar o Vale no **17º dia útil** do mês.
# 2. **Exceção de Novembro / Dezembro**: Em novembro e dezembro, pagar o Vale nos **16º e 17º** dias úteis.
# 3. **Definição de Dia Útil**: Um dia útil é um dia da semana (segunda a sexta) que **não** está marcado como `Feriado = 1` e **não** está marcado como `isClosed = 1`.
# 4. **Nunca no Domingo**: O Vale nunca é pago no domingo. Houve alguns pagamentos no sábado, mas não o suficiente para formar um padrão. Assim, os fins de semana são excluídos da contagem de dias úteis.
# 5. **Tratamento de Feriados / Dias Fechados**: Como a lista de dias úteis já exclui feriados e dias fechados, nenhum ajuste adicional é necessário. As posições 16º/17º são calculadas apenas em dias úteis válidos.
# 6. **Pagamentos Múltiplos**: Apenas novembro e dezembro devem ter dois pagamentos de Vale (16º e 17º dias úteis). Todos os outros meses têm um único pagamento no 17º dia útil.

# %%
def get_vale_date(year, month, df_complete):
    """
    Vale rule based on the true business-day positions:
    - Nov / Dec  → 16th and 17th business days
    - Other months → 17th business day
    """
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date   = start_date + pd.offsets.MonthEnd(0)
    month_dates = pd.date_range(start=start_date, end=end_date)

    date_dict = {
        row['Date']: {'Feriado': row['Feriado'], 'isClosed': row['isClosed']}
        for _, row in df_complete.iterrows()
    }

    biz_days = [
        d for d in month_dates
        if d.weekday() < 5
        and d in date_dict
        and date_dict[d]['Feriado'] != 1
        and date_dict[d]['isClosed'] != 1
    ]

    if month in (11, 12):
        candidates = []
        if len(biz_days) >= 16:
            candidates.append(biz_days[15])   # 16th
        if len(biz_days) >= 17:
            candidates.append(biz_days[16])   # 17th
        return candidates if candidates else None
    else:
        if len(biz_days) >= 17:
            return [biz_days[16]]             # 17th
        print(f"Warning: only {len(biz_days)} business days in {year}-{month:02d}")
        return None


# Test the function
print("Testing Vale dates:")
print("\nJanuary 2018 (should return 16th business day):")
test_date = get_vale_date(2018, 1, df_complete)
if test_date:
    for d in test_date:
        print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")

print("\nNovember 2018 (should return 14th and 15th business days):")
test_date = get_vale_date(2018, 11, df_complete)
if test_date:
    for d in test_date:
        print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")

print("\nDecember 2018 (should return 14th and 15th business days):")
test_date = get_vale_date(2018, 12, df_complete)
if test_date:
    for d in test_date:
        print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")

# %%
# Generate all Vale dates
vale_dates = []
for year in range(2018, 2022):
    for month in range(1, 13):
        dates = get_vale_date(year, month, df_complete)
        if dates:
            for d in dates:
                vale_dates.append({
                    'year': year,
                    'month': month,
                    'vale_date': d,
                    'day_of_month': d.day,
                    'weekday': d.strftime('%A')
                })

vale_dates_df = pd.DataFrame(vale_dates)
print(f"Total Vale dates generated: {len(vale_dates_df)}")
print(vale_dates_df.head(20))

# %% [markdown]
# #### ▪️<span style="color:purple">Note</span>
# 
# Vários meses (especialmente novembro e dezembro, além de alguns outros) tiveram **dois** dias reais de pagamento de Vale.  
# Nossa regra retorna duas datas apenas para nov/dez e uma única data para os demais meses.  
# Em meses de pagamento duplo, a regra captura corretamente pelo menos uma das duas datas reais.  
# Because we only need to flag “Vale day” for previsãoing purposes, capturing one of the two payments is considered a successful prediction.
# 
# 
# Com essa interpretação, a regra atinge uma acurácia prática de **~68%**, suficiente para gerar os dias de Vale futuros.
# 
# |                        | Contagem geral | Removendo dias de Vale duplos que não estão em nov/dez | Após tratar meses de pagamento duplo como sucesso* |
# |------------------------|---------------|-------------------------------------------------|------------------------------------------------|
# | Corresponde à regra           | 30 (64%)      | 30 (68%)                                        | 26 (68%)                                  |
# | Não corresponde             | 17 (36%)      | 14 (32%)                                        | 12 (32%)                                  |
# | Total de dias reais de Vale   | 47            | 44                                              | 38                                        |
# 
# \* Em meses que tiveram dois pagamentos reais, se a regra previu corretamente pelo menos um deles, conta como correspondência.

# %%
df_complete = df_complete.merge(vale_dates_df[['vale_date']], left_on='Date', right_on='vale_date', how='left')

# Add flags
df_complete['is_calculated_vale_day'] = df_complete['Date'] == df_complete['vale_date']
df_complete['vale_matches_rule'] = (df_complete['Vale'] == 1) & (df_complete['is_calculated_vale_day'] == True)

# df_complete

# %%
# Filter and count matches
vale_actual = df_complete[df_complete['Vale'] == 1]  # Actual payments
vale_calculated = df_complete[df_complete['is_calculated_vale_day'] == True]  # Calculated vale days

# Count matches and mismatches
total_vales = len(vale_actual)
matches = len(vale_actual[vale_actual['is_calculated_vale_day'] == True])
mismatches = total_vales - matches

print(f"Total vales: {total_vales}")
print(f"Matches rule: {matches} ({matches/total_vales*100:.1f}%)")
print(f"Mismatches: {mismatches} ({mismatches/total_vales*100:.1f}%)")

# Show mismatches
print("\nMISMATCHES")
mismatch_df = vale_actual[vale_actual['is_calculated_vale_day'] == False]
print(mismatch_df[['Date', 'Vale', 'vale_date']].to_string(index=False))

# Show matches
print("\nMATCHES")
match_df = vale_actual[vale_actual['is_calculated_vale_day'] == True]
print(match_df[['Date', 'Vale', 'vale_date']].to_string(index=False))

# %%
mismatches = [
    "2018-02-26", "2018-04-25", "2018-06-23", "2018-09-25", "2018-12-26",
    "2019-02-23", "2019-03-26", "2019-06-25", "2019-09-26", "2019-10-24",
    "2019-11-27", "2019-12-24", "2020-02-26", "2020-04-25", "2020-05-25",
    "2020-08-24", "2020-10-24", "2020-11-26", "2020-12-24", "2021-01-25", "2021-01-26"
]

print("Predicted vs Real for remaining mismatches:\n")
for d_str in mismatches:
    d = pd.Timestamp(d_str)
    pred = get_vale_date(d.year, d.month, df_complete)
    print(f"{d_str}  →  predicted: {pred}")

# %% [markdown]
# ### ▫️<span style="color:purple">Seção Final Notas</span>
# 
# Agora definimos regras claras que determinam as flags para o futuro:
# - `isClosed`: Provavelmente não haverá vendas nesse dia
# - `Feriado`: Usamos uma API externa para obter feriados brasileiros e determinar quais feriados coincidem com as flags históricas. Criamos uma lista de descrições de feriados para corresponder a datas futuras.
# - `Pagamento`: O evento ocorre a cada 4º dia útil de cada mês.
# - `Vale`: O evento ocorre a cada 17º dia útil de cada mês. Dias 16 adicionais em novembro e dezembro.
# 

# %% [markdown]
# ## 📌<span style="color:blue">Regras de Negócio: Resumo para Períodos Futuros</span>
# 
# Three deterministic rules were derived from the treinamento data so the exogenous flags (`isClosed`, `Pagamento`, `Vale`) can be **reconstructed for any future date** — this is essential because these flags won't exist ahead of time for the holdout period (Jan–Feb/2021) or the March/2021 previsão.
# 
# 
# | Flag                 | Rule (short)                                                                                                                                                                                                                                                    | Match rate on treinamento data                                                                       | Notas                                                                                                                                                                                                                                                                                                                                        |
# | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | `isClosed`           | Loja fechada todo 25 de dezembro e 1º de janeiro                                                                                                                                                                                                                            | 100% (definitional)                                                                               | Determinística, sem exceções encontradas                                                                                                                                                                                                                                                                                                           |
# | `Feriado` (holiday)  | Sinalizar `Feriado = 1` em qualquer data que corresponda a um dos 17 nomes de feriados confirmados, obtidos do dataset externo `holidays.csv`, principalmente feriados Nacionais, além de alguns Estaduais (São Paulo) e Municipais (ex.: cidade de São Paulo) que esse negócio observa | 100% das datas históricas com `Feriado=1` corresponderam a um nome de feriado conhecido (0 de 43 sem descrição) | 11 outros nomes de feriados aparecem no calendário externo, mas nunca foram sinalizados historicamente (feriados regionais fora da área de atuação desse negócio, ex.: datas de Florianópolis/Santa Catarina). Um nome (*"Proc. República Rio Grandense"*) foi sinalizado de forma inconsistente (apenas 2019, não 2018/2020) e foi tratado como NÃO feriado daqui em diante |
# | `Pagamento` (payday) | Pago no 3º ou 4º dia útil do mês (dia útil = seg–sex, excluindo `Feriado` e `isClosed`)                                                                                                                                                   | 67,5% de correspondência exata; 76% quando os dois meses com pagamento duplicado são tratados como um único alvo        | Fev/2018 e dez/2019 tiveram duas datas reais de pagamento; a regra captura uma das duas                                                                                                                                                                                                                                                            |
# | `Vale` (voucher)     | Pago no 17º dia útil do mês; nov/dez também pagam no 16º                                                                                                                                                                                        | 63,8% de correspondência exata; ~68% quando meses de pagamento duplo são tratados como um único alvo                   | Vários meses além de nov/dez também tiveram duas datas reais de Vale; a regra foi intencionalmente simplificada para o padrão dominante                                                                                                                                                                                                                   |
# 
# **Por que isso importa para a modelagem**: 
# - `Feriado`, `Vale` e `Pagamento` são candidatos a regressores exógenos no SARIMAX (Passo F2); 
# - Because the payday/Vale rules only reconstruct ~68–76% of the real event days, the exogenous features used for *future* dates carry known uncertainty, worth flagging explicitly when interpreting previsão error in Steps I/J.
# 

# %% [markdown]
# ## EDA 2: Análise Bivariada

# %% [markdown]
# ### Tamanhos de Efeito das Variáveis Exógenas nas Vendas
# 
# For each exogenous flag (`Feriado`, `Pagamento`, `Vale`), we compute the average sales when the flag is on vs. off, the % change, and whether that difference is statistically significativo, all in one table, instead of scattered print statements.
# Esta é a versão organizada dos números de tamanho de efeito referenciados anteriormente no notebook.
# 
# - Mann-Whitney U test
#     - Hipótese Nula (H₀): as vendas nos dias com flag=0 e flag=1 vêm da mesma distribuição (não há diferença real; qualquer diferença é apenas ruído).
#     - Hipótese Alternativa (H₁): as distribuições diferem — um grupo está genuinamente deslocado para cima ou para baixo.
#     - Decisão rule: p < 0.05 → reject H0 → the flag has a statistically significativo effect on sales.
#     - Escolhido em vez do teste t por não assumir que `Sales_adj` é normalmente distribuída — compara ranks/distribuições em vez de médias.
# 
# 

# %%
def exogenous_effect_summary(data, flags, target='Sales_adj'):
    """
    For each binary flag, compute:
    - mean sales when flag=0 vs flag=1
    - % change (flag=1 vs flag=0)
    - sample sizes
    - Mann-Whitney p-value (flag=1 vs flag=0)

    Returns a tidy summary DataFrame, one row per flag.
    """
    rows = []
    for flag in flags:
        group0 = data.loc[data[flag] == 0, target].dropna()
        group1 = data.loc[data[flag] == 1, target].dropna()

        mean0, mean1 = group0.mean(), group1.mean()
        pct_change = (mean1 - mean0) / mean0 * 100

        stat, p = mannwhitneyu(group1, group0, alternative='two-sided')

        rows.append({
            'Flag': flag,
            'Mean (flag=0)': round(mean0, 2),
            'Mean (flag=1)': round(mean1, 2),
            '% Change': round(pct_change, 1),
            'n (flag=0)': len(group0),
            'n (flag=1)': len(group1),
            'Mann-Whitney p-value': round(p, 4),
            'Significant (α=0.05)': 'Yes' if p < 0.05 else 'No'
        })

    return pd.DataFrame(rows)

flags_to_test = ['Feriado', 'Pagamento', 'Vale']  # add 'isClosed' once its zero-variance issue is fixed
effect_summary = exogenous_effect_summary(df_complete, flags_to_test)
print(effect_summary)

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - Observações:
#     - "% Change" lido junto com o p-valor de Mann-Whitney, que indica se essa diferença é improvável de ser devida ao acaso dados os tamanhos amostrais (colunas *n*). 
# 
#     - This is still an *unconditional* effect (not adjusted for tendência/weekly sazonalidade): a large, significativo % change here is a good candidate for a SARIMAX regressor in Step F2, but the coefficient estimated there (which does control for tendência/sazonalidade) is the more defensible final number.
# 
# - Insights dos Resultados:
#     - **Pagamento** shows a clear, statistically significativo effect (+8.4%, p = 0.0014), sales are reliably higher on payday.
#         - Este é o efeito exógeno com evidência mais forte dos três e um sólido candidato a regressor SARIMAX no Passo F2.
# 
#     - **Feriado** mostra o *maior* efeito bruto (-11,6%), mas **não** atinge significância em α = 0,05 (p = 0,0645) — é limítrofe, não conclusivo. Com apenas 43 dias de feriado nos dados, a diferença poderia ser impulsionada por um punhado de feriados atípicos, e não por um padrão consistente.
#         - O percentual bruto parece convincente, mas o teste estatístico recomenda cautela antes de tratá-lo como um achado firme. Ainda pode valer a pena incluir como regressor dada a forte lógica de negócio (lojas provavelmente fecham ou reduzem horário), mas o próprio coeficiente/p-valor do modelo no Passo F2 deve ser a palavra final, não este teste.
# 
#     - **Vale** mostra um efeito negligenciável (-0,5%, p = 0,5203), sem suporte estatístico. 
#         - Tratá-lo como regressor de baixa prioridade, ou excluí-lo, no Passo F2.
# 
#     - Visão geral: esta tabela é um lembrete útil de que "a maior variação percentual" e "o efeito mais confiável" nem sempre são a mesma variável — `Pagamento` é o achado mais defensável aqui, apesar de ter o menor efeito bruto.

# %% [markdown]
# ### As Flags Exógenas se Sobreõem?
# 
# Os tamanhos de efeito calculados acima (Feriado -11,6%, Pagamento +8,4%, Vale -0,5%) assumiram que cada flag age de forma independente. 
# Se dias de pagamento e feriados frequentemente caem na mesma data, o "efeito feriado" poderia ser parcialmente um efeito de pagamento (ou vice-versa) — isso importa diretamente para a interpretabilidade dos coeficientes SARIMAX no Passo F2.
# 

# %%
flags = ['Feriado', 'Pagamento', 'Vale', 'isClosed']
cooccurrence = df_complete[flags].corr()
print("Correlation between exogenous flags:")
print(cooccurrence.round(3))

# Count of actual overlapping days
for f1, f2 in [('Feriado','Pagamento'), ('Feriado','Vale'), ('Pagamento','Vale')]:
    overlap = ((df_complete[f1] == 1) & (df_complete[f2] == 1)).sum()
    print(f"{f1} & {f2} both = 1 on {overlap} days")

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - As quatro flags mostram correlações fracas entre si (|r| ≤ 0,04), consistentes com quatro eventos de negócio em grande parte independentes e de baixa frequência. Os pequenos valores negativos são o efeito mecânico esperado de combinar várias flags binárias raras, não evidência de uma relação significativa.
#     - Os eventos de `Feriado`, `Pagamento` e `Vale` quase nunca caem na mesma data do calendário, então os tamanhos de efeito individuais (Feriado -11,6%, Pagamento +8,4%, Vale -0,5%) podem ser lidos pelo valor de face — sem confundimento significativo entre eles. Boa notícia para a interpretabilidade no SARIMAX no Passo F2.
# 
# - A coocorrência de `isClosed` com 0 dias de `Feriado`/`Pagamento`/`Vale` **não é um achado independente** — as 7 datas de `isClosed` são exatamente as datas faltantes nos dados originais, então `Feriado`/`Pagamento`/`Vale` foram imputados como 0 por construção (ver nota anterior sobre `fillna(0)`). Essa sobreposição é garantida, não observada.
# 
# - A matriz de correlação em si, porém, é um resultado legítimo (não um artefato da imputação) — reflete as baixas taxas-base das quatro flags.
# 
# - Because the flags don't meaningfully co-occur, the individual effect-size estimates in the earlier bivariate analysis (`Feriado`, `Pagamento`, `Vale` vs. `Sales_adj`) can be interpreted without worrying about confounding *between these flags*, though confounding with the underlying tendência/sazonalidade (not yet tested) is still possible and is addressed properly once SARIMAX controls for it in Step F2.

# %% [markdown]
# ### Vendas vs. Dia da Semana
# 
# Before treating this as "weekly sazonalidade," it's worth establishing it here as a business relationship: does average sales genuinely differ by weekday, or could the differences plausibly be due to chance?
# 
# - **Kruskal-Wallis** (ANOVA não paramétrica) testa se pelo menos a distribuição de um dia da semana difere das demais — apropriado aqui, pois as vendas diárias não são garantidamente normais.
#     - Hipótese Nula (H₀): Todas as medianas populacionais dos grupos (ou ranks médios) são iguais.
#     - Hipótese Alternativa (H₁): At least one group population median is significativoly different.
#     - Decisão rule: p < 0.05 → reject H0 → at least one weekday avg sales is statistically significativo from the others.
# 

# %%
df_complete['Weekday'] = df_complete['Date'].dt.day_name()
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_complete, x='Weekday', y='Sales_adj', order=weekday_order, ax=ax)
ax.set_title('Sales Distribution by Day of the Week')
plt.tight_layout(); plt.show()

groups = [df_complete.loc[df_complete['Weekday'] == d, 'Sales_adj'].dropna() for d in weekday_order]
stat, p = kruskal(*groups)
print(f"Kruskal-Wallis H = {stat:.2f}, p-value = {p:.4g}")

# %%
# Ensure correct weekday order
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_chart1 = df_complete.copy()
df_chart1['Weekday'] = pd.Categorical(df_chart1['Weekday'], categories=weekday_order, ordered=True)
df_chart1['Year-Week'] = df_complete['Date'].dt.strftime('%Y-%V')
df_chart1 = df_chart1.sort_values(['Year-Week', 'Weekday'])

# Create line chart with all weeks
plt.figure(figsize=(14, 10))
sns.set_style("whitegrid")

# Plot each week as a separate line
for week in df_chart1['Year-Week'].unique():
    week_data = df_chart1[df_chart1['Year-Week'] == week]
    sns.lineplot(data=week_data, x='Weekday', y='Sales_adj', 
                 marker='o', markersize=8, linewidth=2.5, 
                 label=week)

plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
plt.ylabel('Sales ($)', fontsize=12, fontweight='bold')
plt.title('Sales Trend by Day of Week - All Weeks', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Year-Week')
# plt.legend([]) # no legend
# Place legend at bottom
plt.legend(bbox_to_anchor=(0.5, -0.15),  # Position at bottom center
           loc='upper center',           # Anchor point
           ncol=12,                      # Number of columns
           fontsize=10,
           title='Year-Week')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - Os valores de vendas de sábado são geralmente mais altos que o restante da semana.
# 
# - With a p-value ~ 0, we can say that at least one weekday is different than the others. A significativo result that backs up treating weekday as a structural driver, not noise, going into Step A/E.

# %% [markdown]
# ### Vendas vs. Mês
# 
# Same logic as weekday, but for month, this is a first, non-time-series look at whether "month" as a business/calendar category relates to sales, independent of any sazonal-decomposition assumptions.
# 

# %%
df_complete['Month_Name'] = df_complete['Date'].dt.month_name()
month_order = ['January','February','March','April','May','June','July',
               'August','September','October','November','December']

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df_complete, x='Month_Name', y='Sales_adj', order=month_order, ax=ax)
ax.set_title('Sales Distribution by Month')
plt.xticks(rotation=45)
plt.tight_layout(); plt.show()

groups = [df_complete.loc[df_complete['Month_Name'] == m, 'Sales_adj'].dropna() for m in month_order]
stat, p = kruskal(*groups)
print(f"Kruskal-Wallis H = {stat:.2f}, p-value = {p:.4g}")

# %%
# Prepare data for monthly aggregation
df_chart2 = df_complete.copy()

# Create year and month columns
df_chart2['Year'] = df_chart2['Date'].dt.year
df_chart2['Month'] = df_chart2['Date'].dt.month
df_chart2['Month_Name'] = df_chart2['Date'].dt.strftime('%b')  # Jan, Feb, Mar, etc.

# Aggregate sales by Year and Month (SUM)
monthly_sales = df_chart2.groupby(['Year', 'Month', 'Month_Name'])['Sales_adj'].sum().reset_index()

# Sort by month number
monthly_sales = monthly_sales.sort_values(['Year', 'Month'])

# Create the line chart
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# Plot each year as a separate line
years = monthly_sales['Year'].unique()
for year in years:
    year_data = monthly_sales[monthly_sales['Year'] == year]
    plt.plot(year_data['Month_Name'], year_data['Sales_adj'], 
             marker='o', linewidth=2.5, markersize=8, 
             label=str(year))

plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Total Sales ($)', fontsize=12, fontweight='bold')
plt.title('Monthly Sales Comparison Across Years (sum)', fontsize=14, fontweight='bold')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Prepare data for monthly aggregation
df_chart3 = df_complete.copy()

# Create year and month columns
df_chart3['Year'] = df_chart3['Date'].dt.year
df_chart3['Month'] = df_chart3['Date'].dt.month
df_chart3['Month_Name'] = df_chart3['Date'].dt.strftime('%b')  # Jan, Feb, Mar, etc.

# Aggregate sales by Year and Month (AVG)
monthly_sales = df_chart3.groupby(['Year', 'Month', 'Month_Name'])['Sales_adj'].mean().reset_index()

# Sort by month number
monthly_sales = monthly_sales.sort_values(['Year', 'Month'])

# Create the line chart
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# Plot each year as a separate line
years = monthly_sales['Year'].unique()
for year in years:
    year_data = monthly_sales[monthly_sales['Year'] == year]
    plt.plot(year_data['Month_Name'], year_data['Sales_adj'], 
             marker='o', linewidth=2.5, markersize=8, 
             label=str(year))

plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Total Sales ($)', fontsize=12, fontweight='bold')
plt.title('Monthly Sales Comparison Across Years (avg)', fontsize=14, fontweight='bold')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - Dezembro parece ter vendas mais altas que o restante do ano; fevereiro similar, mas em menor escala.
# 
# - Since there are only ~3 observations of each month (one per year), treat any "significativo" result cautiously, it's consistent with, but doesn't prove, a genuine annual sazonal pattern (cross-reference against the year-over-year overlay chart discussed earlier).
# 
# - **Teste Kruskal-Wallis**: Com p-valor ~ 0, podemos dizer que pelo menos um mês é diferente dos outros.
# 
# - **Question**
#     - _Can we create a monthly sazonal index to capture this sazonalidade to not only rely on the weekly tendência?_

# %%
# Monthly seasonal index: average sales per month, normalized to a baseline of 1.0
monthly_avg = df_complete.groupby(df_complete['month'])['Sales_adj'].mean()
overall_avg = df_complete['Sales_adj'].mean()

seasonal_index = (monthly_avg / overall_avg).round(3)
seasonal_index_byMonth = seasonal_index.copy()
seasonal_index_byMonth.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(seasonal_index_byMonth)

fig, ax = plt.subplots(figsize=(10,4))
seasonal_index_byMonth.plot(kind='bar', ax=ax)
ax.axhline(1.0, color='red', linestyle='--', label='Baseline (no seasonal effect)')
ax.set_title('Monthly Seasonal Index (1.0 = average month)')
ax.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ### Vendas vs. Todas as Flags: Correlação Consolidada
# 
# Uma única matriz de correlação, tratando as flags binárias como numéricas 0/1 junto com `Sales_adj`, oferece uma visão consolidada de todas as relações bivariadas cobertas até agora (flag vs. flag e agora flag vs. alvo) em um só lugar.
# 
# - A correlação ponto-bisserial (Pearson em flag 0/1) é um resumo linear razoável aqui, mas tenha em mente que pode subestimar um efeito real se a relação não for linear — cruzar com os resultados de boxplot/Mann-Whitney em vez de confiar apenas nesta tabela.

# %%
corr_vars = ['Sales_adj', 'Feriado', 'Pagamento', 'Vale', 'isClosed'] 

corr_matrix = df_complete[corr_vars].corr()
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation: Sales vs. Exogenous Flags')
plt.tight_layout(); plt.show()

# %% [markdown]
# 
# #### ▪️<span style="color:purple">Notas</span>
# 
# - `isClosed` (-0,220) é de longe a relação mais forte da tabela: esperado, pois as vendas são ~0 nos 7 dias em que a loja não opera. Isso confirma a verificação anterior de qualidade dos dados, não é um achado novo.
# - `Feriado` (-0.059) is weak, consistent with the earlier Mann-Whitney result (p=0.0645, not significativo at α=0.05).
# - `Pagamento` (+0.047) looks small next to isClosed, but should **not** be read as "Pagamento barely matters": it was the one flag confirmed statistically significativo (p=0.0014) in the earlier bivariate test. Correlation coefficients are compressed by low base-rate flags (Pagamento is "on" only ~3.5% of days), so a real, consistent effect on those days still produces a modest Pearson r across the full dataset. Effect size + significance (from the earlier table) is the more reliable read than the raw correlation coefficient alone.
# - `Vale` (0.001) shows no linear relationship, consistent with its earlier non-significativo result (p=0.52).
# - Conclusão prática: correlation strength here should be read alongside the earlier Mann-Whitney effect-size table, not in isolation: `isClosed` and `Pagamento` are the two flags with genuine, reliable relationships to sales, despite very different correlation magnitudes.

# %% [markdown]
# ### ▫️<span style="color:purple">Seção Final Notas</span>
# 
# - The monthly sazonal index explored above (Dec 1.184, Nov 1.043, Apr 0.942, etc.) was computed using the full date range available at this point in the notebook, which includes the holdout period (Jan–Feb/2021). Using it as-is would leak holdout information into a feature used later for modeling.
# 
# - **Decisão**: `Seasonal_Index` será adicionada como coluna a `df_complete` apenas após a divisão treino/holdout do Passo B, calculada apenas de `ts_train` e depois mapeada para todas as linhas (treino, holdout e futuro). Isso evita qualquer vazamento de holdout, mantendo a feature disponível para a comparação SARIMAX planejada no Passo F2.
# 
# - **Resumo: variáveis ainda necessárias para o dataset de previsão futura**: A Parte 1 derivou várias regras e features dos dados históricos, mas nenhuma delas existe ainda para datas futuras (ex.: março/2021). A tabela abaixo consolida tudo o que deve ser reconstruído quando o dataset futuro for construído no Passo J.
# 
# | Variable | Regra de origem | Status |
# |---|---|---|
# | `isClosed` | Fechada todo 25 de dezembro e 1º de janeiro (verificação determinística de data) | Regra definida, pronta para aplicar |
# | `Feriado` | Corresponder a nomes de feriados conhecidos de `holidays.csv` (17 nomes confirmados) | Regra definida, pronta para aplicar |
# | `Pagamento` | 3º ou 4º dia útil do mês (dia útil = seg–sex, excluindo `Feriado`/`isClosed`) | Regra definida (`get_payment_date()`), ~68–76% de acurácia histórica |
# | `Vale` | 17º dia útil do mês; também 16º em nov/dez | Regra definida (`get_vale_date()`), ~63,8–68% de acurácia histórica |
# | `Seasonal_Index` | Índice mensal (Dez 1,184, Nov 1,043, Abr 0,942 etc.) | Adiado — calculado apenas de `ts_train` (após divisão do Passo B) e mapeado para datas futuras por mês, conforme a decisão acima |
# 
# - `Pagamento` and `Vale` are the two flags with imperfect reconstruction rules, any previsão for a future period inherits that uncertainty, since the model will be conditioned on *predicted* rather than *actual* event days. Worth restating this limitation when interpreting previsão error in Step I/J.

# %% [markdown]
# # 🔹**PARTE 2**: Análise de Séries Temporais
# 
# ---

# %% [markdown]
# ## 🔸Passo A: Análise Visual da Série
# 
# Reunindo a análise visual/estatística completa do Passo A (gráficos originais + estabilidade de variância, ACF de longo alcance, heatmap de calendário, sobreposição ano a ano, scatterplots de defasagem) em uma visão consolidada antes de passar ao Passo B.
# 
# - Full Series
# - Primeiros 90 dias
# - Últimos 90 dias
# - Decomposição Aditiva
# - Decomposição Multiplicativa
# 

# %%
# full series
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_complete, x='Date', y='Sales_adj')
plt.title(f"Sales Over Time: {choose_dept}")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# %%
# First 90 days
period_90f = df_complete.head(90).copy()

plt.figure(figsize=(14, 8))
sns.lineplot(data=period_90f, x='Date', y='Sales_adj')

# Highlight weekends (Saturday=5, Sunday=6)
for date in period_90f['Date'].unique():
    if date.weekday() >= 5:  # Saturday or Sunday
        plt.axvspan(date - pd.Timedelta(hours=12), 
                   date + pd.Timedelta(hours=12), 
                   alpha=0.1, color='blue')

plt.title(f"Sales Over Time: {choose_dept} - first 90 days")
plt.xlabel("Date")
plt.ylabel("Adjusted Sales")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Last 90 days
period_90l = df_complete.tail(90).copy()

plt.figure(figsize=(14, 8))
sns.lineplot(data=period_90l, x='Date', y='Sales_adj')

# Highlight weekends (Saturday=5, Sunday=6)
for date in period_90l['Date'].unique():
    if date.weekday() >= 5:  # Saturday or Sunday
        plt.axvspan(date - pd.Timedelta(hours=12), 
                   date + pd.Timedelta(hours=12), 
                   alpha=0.1, color='blue')

plt.title(f"Sales Over Time: {choose_dept} - first 90 days")
plt.xlabel("Date")
plt.ylabel("Adjusted Sales")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Decomposição Aditiva vs. Multiplicativa
# 
# Comparing both decomposition models side-by-side helps answer the open Step A question of whether tendência/sazonal effects add a constant amount or scale with the level. Multiplicative decomposition requires strictly positive values, so any exact zeros (e.g., on `isClosed` days) are replaced with a tiny epsilon before decomposing — this only affects the multiplicative version's ability to run, not the underlying data.
# 

# %%
# Ensure ts exists before decomposing
ts = df_complete.set_index('Date')['Sales_adj']

# Replace exact zeros with a small epsilon so multiplicative decomposition can run
ts_for_decomp = ts.copy()
ts_for_decomp = ts_for_decomp.where(ts_for_decomp != 0, 0.000001)

print(f"Zeros replaced: {(ts == 0).sum()}")
print(f"Remaining non-positive values: {(ts_for_decomp <= 0).sum()}")

decomp_add = seasonal_decompose(ts_for_decomp, model='additive', period=7)
decomp_mul = seasonal_decompose(ts_for_decomp, model='multiplicative', period=7)

fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)

for col, decomp, title in [(0, decomp_add, 'Additive'), (1, decomp_mul, 'Multiplicative')]:
    axes[0, col].plot(decomp.observed); axes[0, col].set_title(f'{title} — Observed')
    axes[1, col].plot(decomp.trend); axes[1, col].set_title(f'{title} — Trend')
    axes[2, col].plot(decomp.seasonal); axes[2, col].set_title(f'{title} — Seasonal')
    axes[3, col].plot(decomp.resid); axes[3, col].set_title(f'{title} — Residual')
    for row in range(4):
        axes[row, col].grid(alpha=0.3)

plt.tight_layout(); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

zoom_start, zoom_end = '2020-01-01', '2020-03-31'

axes[0].plot(decomp_add.seasonal.loc[zoom_start:zoom_end])
axes[0].set_title('Additive Seasonal (zoomed)')
axes[0].grid(alpha=0.3)

axes[1].plot(decomp_mul.seasonal.loc[zoom_start:zoom_end])
axes[1].set_title('Multiplicative Seasonal (zoomed)')
axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - The epsilon replacement (0 → 0.000001) only affects the 7 `isClosed` days (and any other exact-zero rows), small enough to not distort the rest of the series, but expect a visible spike/artifact in the multiplicative sazonal or residual component right at those exact dates, since dividing by a near-zero tendência estimate can blow up the ratio locally.
# 
# - As linhas Observed e Trend são visualmente idênticas entre os dois modelos
#     - expected, since the tendência estimate itself doesn't depend on additive/multiplicative choice.
# 
# - Seasonal: additive shows an absolute (~-0.5M to +1.25M) adjustment; multiplicative shows a proportional (~0.8x to 1.6x) adjustment. Neither row can show whether sazonal amplitude scales with the tendência level, since classical decomposition assumes a fixed sazonal pattern throughout.
#     - Short period chart show a dense, blocky pattern because the 7-day ciclo is compressed across ~3 years of x-axis, a zoomed-in view is more readable for confirming the actual weekly shape.
# 
# - O residual é o fator decisivo: o **residual aditivo cresce substancialmente ao longo do tempo** (maiores oscilações em 2020–2021, até ±2–3M), indicando que o modelo aditivo deixa variância não explicada crescente conforme as vendas crescem. O **residual multiplicativo permanece em uma faixa comparável ao longo de todo o período**, consistente com a variância escalando proporcionalmente ao nível.
# 
# - **Conclusão**: 
#     - multiplicative decomposition is the better fit for this series. This carries forward as the working choice for Step F1 (Holt-Winters: `tendência='mul'`, `sazonal='mul'`) and suggests modeling `log(Sales_adj)` may also be worth testing in Step D/F2, since a log transform turns a multiplicative relationship into an additive one that ARIMA/SARIMA can work with directly.
# 
# 
# - **Signals**:
#     - **Trend**: Shows tendências, increasing through the years.
#     - **Seasonality**: Weekly sazonalidade detected (7 days).
#     - **Cicles**: Sem ciclos visíveis.

# %% [markdown]
# ## 🔸Passo B: Treino / Holdout 
# 
# **Objective**: split the series into a treinamento set (through Dec/2020) and a holdout set (Jan–Feb/2021), matching the Descrição dos Dados. From this point forward, all identification, estimation, and diagnostics (Steps C–H) use `ts_train` only, the holdout is untouched until Step I.
# 
# **Note**: treinamento data represents "known" historical performance used to fit the model; holdout simulates "future" data to validate accuracy before it's trusted to previsão March/2021.

# %%
# Define dates for the split
train_end = '2020-12-31'
holdout_start = '2021-01-01'
holdout_end = '2021-02-28'

# %% [markdown]
# ### ➕Aplicando a Substituição por Epsilon
# 
# Como `df_complete` carrega as variáveis exógenas junto com as vendas (necessárias para o SARIMAX do Passo F2), a substituição por epsilon é aplicada aqui diretamente em `Sales_adj`, e não apenas na série isolada `ts`. Isso mantém `ts`/`ts_train`/`ts_holdout` e `exog_train`/`exog_holdout` construídos a partir da mesma fonte consistente daqui em diante.
# 

# %%
EPSILON = 0.000001

n_zero_before = (df_complete['Sales_adj'] == 0).sum()
df_complete['Sales_adj'] = df_complete['Sales_adj'].where(df_complete['Sales_adj'] != 0, EPSILON)
n_zero_after = (df_complete['Sales_adj'] == 0).sum()

print(f"Zeros before: {n_zero_before}, zeros after: {n_zero_after}")
print(f"Min value: {df_complete['Sales_adj'].min():.6f}")


# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - Substituição agora aplicada na fonte (`df_complete['Sales_adj']`), não apenas na série derivada `ts` — isso significa que `exog_train`/`exog_holdout` construídos depois a partir de `df_complete` para o Passo F2 permanecerão consistentes com `ts_train`/`ts_holdout`.
# - `ts`, `ts_train`, `ts_holdout` são reconstruídos a partir de `df_complete` após a substituição, usando os mesmos limites `train_end`/`holdout_start`/`holdout_end` do início do Passo B.
# - Confirmar que restam 0 zeros / 0 negativos nas três séries.
# - Trade-off a notar: `df_complete['Sales_adj']` não preserva mais o fato literal de "0 vendas nos dias isClosed" diretamente nessa coluna — mas `isClosed` permanece como flag separada, então essa informação não se perde, apenas é representada de forma diferente (flag + valor próximo de zero em vez de flag + zero exato).

# %% [markdown]
# ### ➕Adicionando Índice de Sazonalidade por Mês
# 
# Let's recalculate and add the sazonalidade index by month using only train dataset.
# 

# %%
df_complete_train = df_complete[df_complete['Date'] <= train_end]

monthly_avg_train = df_complete_train.groupby(df_complete['month'])['Sales_adj'].mean()
overall_avg_train = df_complete_train['Sales_adj'].mean()

seasonal_index_train = (monthly_avg_train / overall_avg_train).round(3)
print('Train Dataset Seasonality Index:')
print(seasonal_index_train)

# %%
print('Full Dataset Seasonality Index:')
print(seasonal_index)

# %%
# Adding seasonality index to the main DF
df_complete['Seasonal_Index'] = df_complete['Date'].dt.month.map(seasonal_index_train)

print(df_complete[['Date', 'Seasonal_Index']].head())
print(f"\nMissing values: {df_complete['Seasonal_Index'].isna().sum()}")

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# Comparação entre o índice do dataset completo vs. dataset de treino.
# 
# - A maioria dos meses (mar–dez) muda apenas ligeiramente (~+0,01–0,02) entre o índice do dataset completo e o apenas-treino.
#     - ruído esperado por excluir 2 de 38 meses.
# 
# - Jan e fev mudam de forma mais substancial: jan cai de 0,955 para 0,883, e fev passa de *acima* da baseline (1,023) para *abaixo* da baseline (0,957). 
#     - Isso indica que jan–fev/2021 (o período de holdout) teve desempenho atipicamente forte em relação ao padrão típico de 2018–2020 para esses meses.
# 
# - Isso é uma boa validação da abordagem apenas-treino: se o índice do dataset completo tivesse sido usado, o `Seasonal_Index` de fev teria sido informado pelos próprios dados usados para julgar a acurácia do holdout no Passo I, e teria superestimado o quão forte fevereiro tipicamente é.
# 
# - Practical implication: expect the holdout evaluation (Step I) to show the model under-predicting Jan/Feb 2021 sales somewhat, since the train-only sazonal index treats those months as below-average, while the actual holdout data was atypically strong.
# 
# 
# | Month_nbr | Month | Full Dataset | Treino Dataset | Delta  |
# | --------- | ----- | ------------ | ------------- | ------ |
# | 1         | Jan   | 0.955        | 0.883         | -0.072 |
# | 2         | Feb   | 1.023        | 0.957         | -0.066 |
# | 3         | Mar   | 0.999        | 1.011         | 0.012  |
# | 4         | Apr   | 0.942        | 0.954         | 0.012  |
# | 5         | May   | 0.972        | 0.984         | 0.012  |
# | 6         | Jun   | 0.965        | 0.977         | 0.012  |
# | 7         | Jul   | 0.955        | 0.967         | 0.012  |
# | 8         | Aug   | 0.978        | 0.99          | 0.012  |
# | 9         | Sep   | 0.977        | 0.99          | 0.013  |
# | 10        | Oct   | 1.014        | 1.027         | 0.013  |
# | 11        | Nov   | 1.043        | 1.056         | 0.013  |
# | 12        | Dec   | 1.184        | 1.199         | 0.015  |
# 
# 
# 

# %% [markdown]
# ### → Divisão Treino / Holdout
# 
# Por que essa divisão importa para o negócio:
# 
# - Treinoing data represents the "known" historical performance used to build the model
# 
# - Os dados de holdout (jan-fev 2021) simulam dados "futuros" para validar a acurácia do modelo antes da implantação
# 
# - This approach ensures our model will perform well on real future data (March 2021 previsão)

# %%
ts = df_complete.set_index('Date')['Sales_adj'] 

ts_train = ts.loc[:train_end]
ts_holdout = ts.loc[holdout_start:holdout_end]

print(f"Train:   {ts_train.index.min().date()} to {ts_train.index.max().date()}  ({len(ts_train)} obs)")
print(f"Holdout: {ts_holdout.index.min().date()} to {ts_holdout.index.max().date()}  ({len(ts_holdout)} obs)")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_train.index, ts_train.values, label='Train', linewidth=0.8)
ax.plot(ts_holdout.index, ts_holdout.values, label='Holdout', color='orange', linewidth=1.2)
ax.axvline(pd.Timestamp(train_end), color='red', linestyle='--', label='Split point')
ax.legend(); ax.set_title('Train / Holdout Split'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 🔸Passo C: Verificação de Ruído Branco
# 
# **Objective**: confirm `ts_train` is not ruído branco before investing further in a model — if it were, no model could beat a naive mean/last-value previsão, and Steps D–J would be moot.

# %% [markdown]
# ### Teste de Ljung-Box
# 
# - H0: Series is ruído branco (no autocorrelation at the tested lags). 
# - H1: tem estrutura explorável.
# - Decisão rule: p-valor < 0,05 em qualquer defasagem → rejeitar H0.

# %%
lb_test = acorr_ljungbox(ts_train, lags=[7, 14, 21, 30], return_df=True)
print(lb_test)

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# We ran the Ljung-Box test for different lags to check if we would find a ruído branco on the most commom lags on a daily series.
# - Em todas as defasagens, p-valor < 0,05, rejeitando a hipótese nula.
# - The series is not a ruído branco, there are clearly patterns that we can predict with a time series model.
# 

# %% [markdown]
# ### ACF Plot for Verificação de Ruído Branco
# 
# Um complemento visual ao teste de Ljung-Box — uma série de ruído branco mostraria (quase) todas as barras dentro da banda de confiança.

# %%
fig, ax = plt.subplots(figsize=(14, 4))
plot_acf(ts_train.dropna(), lags=40, ax=ax, alpha=0.05)
ax.set_title('ACF of Training Series: White Noise Check')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - A defasagem 7 (e múltiplos de 7) são as que caem fora da banda de confiança, mostrando que é um sinal dominante também no dataset de treino.
# - Outras defasagens 3, 4, 5 também caem fora da banda de confiança, mas estão bem próximas, enquanto a defasagem 7 é 3/4 vezes maior.
# 

# %% [markdown]
# ## 🔸Passo D: Verificação de Estacionariedade
# 
# **Objective**: determine whether `ts_train` is stationary, and if not, how much differencing (`d`, sazonal `D`) is needed before ARIMA/SARIMA (Step F2) can be applied. Exponential smoothing (Step F1) doesn't require stationarity, but this still matters for choosing between the two families.
# 

# %% [markdown]
# ### Teste de Dickey-Fuller Aumentado (ADF)
# 
# - H0: a série tem raiz unitária (não estacionária). 
# - H1: a série é estacionária.
# - Decisão rule: p-valor < 0,05 → rejeitar H0 → estacionária.

# %%
def run_adf(series, label=''):
    result = adfuller(series.dropna())
    print(f"ADF Test: {label}")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    for k, v in result[4].items():
        print(f"  Critical Value ({k}): {v:.4f}")
    conclusion = 'Stationary' if result[1] < 0.05 else 'Non-stationary'
    print(f"  Conclusion: {conclusion} (alpha = 0.05)\n")
    return result

run_adf(ts_train, 'Sales_adj (level)')

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - **Result**: Estatística ADF = -1,6217, p-valor = 0,4718 → falha em rejeitar H0 → **não estacionária** no nível.
# 
# - This is expected given Step A's confirmed upward tendência, a non-constant mean is exactly what the ADF test is picking up on. 
#     - Diferenciação é necessária antes do Passo F2 (ARIMA/SARIMA);
#     - O Passo F1 (suavização exponencial) ainda pode trabalhar diretamente na série em nível, pois não exige estacionariedade.

# %% [markdown]
# ### Diferenciação se Necessário
# 
# If the level series is non-stationary, test regular and sazonal differencing (s=7, from the weekly sazonalidade confirmed in Step A).

# %%
ts_diff1 = ts_train.diff().dropna()
run_adf(ts_diff1, 'after 1st regular difference (d=1)')

ts_seasdiff = ts_train.diff(7).dropna()
run_adf(ts_seasdiff, 'after seasonal difference (D=1, s=7)')

ts_diff1_seasdiff = ts_train.diff().diff(7).dropna()
run_adf(ts_diff1_seasdiff, 'after regular + seasonal difference (d=1, D=1, s=7)')

# %% [markdown]
# ### Escolhendo Entre d=1, D=1 e d=1+D=1
# 
# All three differenced series pass ADF, so the ADF test alone can't decide, it detects unit roots, not sazonalidade. 
# O ACF de cada candidato abaixo mostra se o padrão semanal (defasagem 7) ainda está presente após cada escolha de diferenciação, e se aparecem artefatos de sobrediferenciação (um pico negativo acentuado na defasagem 1).
# 
# - Insight principal: qual escolha de diferenciação remove o pico da defasagem 7 sem introduzir artefato de sobrediferenciação?

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

plot_acf(ts_diff1, lags=30, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF after d=1 only: is the lag-7 spike still there?')
axes[0].grid(alpha=0.3)

plot_acf(ts_seasdiff, lags=30, ax=axes[1], alpha=0.05)
axes[1].set_title('ACF after D=1 only (s=7): is there a trend-like slow decay?')
axes[1].grid(alpha=0.3)

plot_acf(ts_diff1_seasdiff, lags=30, ax=axes[2], alpha=0.05)
axes[2].set_title('ACF after d=1 + D=1: cleanest, or over-differenced (sharp negative lag-1)?')
axes[2].grid(alpha=0.3)

plt.tight_layout(); plt.show()

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - **d=1 only** still shows a spike at lag 7 (regular differencing removes tendência, not weekly sazonalidade).
# - **D=1 only** still shows a slower decay pattern (sazonal differencing removes the weekly ciclo, but a residual tendência can remain).
# - **d=1 + D=1** shows neither, the standard choice for a series with both a confirmed tendência and confirmed weekly sazonalidade (Step A). Watch for a large negative spike at lag 1, which would indicate over-differencing.
# 
# 
# ### Determinar os Parâmetros d e D
# 
# - **d = 0** (nenhuma diferenciação regular necessária)
# - **D = 1**, **s = 7** (sazonal differencing only)
# 
# **Reasoning**:
# - `d=1` alone leaves the weekly sazonalidade completely intact (clear spikes at lags 7, 14, 21, 28), regular differencing removes tendência, not the weekly ciclo.
# - `D=1` sozinho alcança estacionariedade (ADF) e produz o ACF mais limpo dos três candidatos — a maioria das defasagens cai dentro da banda de confiança, restando apenas um pico suave na defasagem 1 (~0,22) e na defasagem 7 (~-0,3).
# - `d=1 + D=1` combinados mostram um forte pico negativo na defasagem 1 (-0,48), assinatura clássica de sobrediferenciação, consistente com a estatística ADF *mais fraca* do caso combinado (-12,59) vs. apenas d=1 (-14,61).
# - The two remaining spikes in the D=1-only panel (mild lag-1, lag-7) are informative rather than a problem, they'll help identify the AR/MA and sazonal AR/MA terms (p, q, P, Q) in Step F2, using this same differenced series.

# %% [markdown]
# ## 🔸Passo E: Identificação de Componentes (tendência, sazonalidade, ciclo)
# 
# **Objective**: formally consolidate what's already been established (Steps A, B, D) into named components (tendência, sazonalidade, ciclo) and confirm the one open question: does the Nov/Dec effect repeat every year, or was it driven by a single year?
# 

# %% [markdown]
# ### Trend
# 
# Already established in Passo A: sustained growth (+43.7% 2018→2021 | +32.8% 2018→2020), and the additive-vs-multiplicative decomposition showed the multiplicative model's residual stayed more stable over time, supporting a tendência whose *absolute* size grows alongside the level, not a fixed dollar amount per period.

# %%
yearly_avg = df_complete[df_complete['Date'] <= train_end].groupby('year')['Sales_adj'].mean()
print(yearly_avg)

first_year, last_year = yearly_avg.index.min(), yearly_avg.index.max()
growth_pct = (yearly_avg[last_year] - yearly_avg[first_year]) / yearly_avg[first_year] * 100

print(f"\nAvg daily sales {first_year}: {yearly_avg[first_year]:,.2f}")
print(f"Avg daily sales {last_year}: {yearly_avg[last_year]:,.2f}")
print(f"Growth: {growth_pct:+.1f}%")

# Caveat check: is the last year a full year, or partial?
days_in_last_year = df_complete[(df_complete['Date'] <= train_end) & (df_complete['year'] == last_year)].shape[0]
print(f"\nDays counted in {last_year}: {days_in_last_year} (full year = 365/366)")

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - **Result**: A média de vendas diárias cresceu de **R$ 1,8M** (2018) para **R$ 2,4M** (2020): aumento de **+32,8%**, comparando dois anos calendário completos (365 e 366 dias, respectivamente).
# 
# - A análise do Passo B já mostrou que jan–fev/2021 foi atipicamente forte em relação ao padrão típico.
# 
# - 2019 sits between the two ($2M), suggesting fairly steady year-over-year growth rather than one anomalous jump, consistent with the smooth upward tendência line seen in Step A's plots.
# 

# %% [markdown]
# ### Análise do Componente Sazonal
# 
# Weekly sazonalidade (`s=7`) is already confirmed (Step A's ACF, Step D's ADF/ACF on the D=1 differenced series). 
# What's still open: is the Nov/Dec pattern (Seasonal_Index 1.056/1.199 from Step B) a genuine annual sazonalidade, or does it come from just one or two unusual years?

# %%
df_complete['Year'] = df_complete['Date'].dt.year

monthly_by_year = df_complete[df_complete['Date'] <= train_end].groupby(
    ['Year', 'month']
)['Sales_adj'].mean().unstack(level=0)

fig, ax = plt.subplots(figsize=(12, 5))
monthly_by_year.plot(ax=ax, marker='o')
ax.set_title('Monthly Average Sales by Year (Train Only): Is Nov/Dec Consistent?')
ax.set_xlabel('Month'); ax.set_ylabel('Average Sales_adj')
ax.legend(title='Year')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - December rises in all three treinamento years (2018, 2019, 2020), a consistent pattern across years supports treating `Seasonal_Index` as a genuine sazonal component rather than a one-off.
# 
# - *Caveat*: with only 3 years of treinamento data, this is 3 observations per month, enough to spot a gross inconsistency, but not enough to rule out a subtler year-over-year drift.
# 

# %% [markdown]
# ### Cycle
# 
# Cycle refers to fluctuations longer than a year that aren't tied to a fixed calendar period (e.g., multi-year business ciclos).
# With only ~3 years of treinamento data, this is difficult to separate from tendência.
# 

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - With 3 years of data, a ciclo (which by definition operates on a timescale longer than a year and isn't fixed-period) can't be reliably distinguished from the tendência itself. 
# 
# - **Reasonable conclusion**: no ciclo component is modeled separately. Its effects, if any, are absorbed into the tendência term.
# 
# ---

# %% [markdown]
# ### Resumo da Identificação de Componentes de Negócio
# 
# | Component | Achado | Evidência |
# |---|---|---|
# | **Trend** | Aumento forte e sustentado (+32,8%, 2018→2020); de natureza multiplicativa | Gráfico de série temporal do Passo A, comparação de residual da decomposição |
# | **Seasonality: Weekly** | Confirmada, `s=7`; pico no sábado / vale no domingo (razão 2,24x) | ACF/heatmap do Passo A, ACF do Passo D na série diferenciada D=1 |
# | **Seasonality: Monthly/Annual** | Concentrada em nov/dez (`Seasonal_Index` 1,056/1,199) | Step B sazonal index, this section's year-over-year chart |
# | **Cycle** | Not modeled separately — indistinguishable from tendência given only 3 years of data | Esta seção |
# | **Exogenous Effects** | `Pagamento` significativo (+8.4%, p=0.0014); `Feriado` borderline (-11.6%, p=0.0645); `Vale` not significativo | Análise bivariada da Parte 1 (Mann-Whitney) |
# 
# 
# **Implicação para o Passo F**: a série exige
# 1. a model that handles tendência and weekly sazonalidade, SARIMA with `D=1, s=7` (Step D) or Holt-Winters with multiplicative tendência/sazonal (Step A decomposition) and
# 2. `Pagamento` como candidato principal a regressor exógeno, com `Feriado` como secundário que vale testar apesar da significância limítrofe.

# %% [markdown]
# ## 🔸Passo F: Ajuste de Modelos
# 
# **Model selection based on components identified in Step E**: the series has both a tendência and weekly sazonalidade, ruling out Simple Suavização Exponencial (no tendência/season) and Holt's method (tendência only). 

# %% [markdown]
# ### Justificando a Escolha de Holt-Winters
# 
# Before jumping to Holt-Winters, let's confirm (rather than assume) that the simpler exponential smoothing methods aren't sufficient. 
# **Simple Suavização Exponencial** (no tendência/season) and **Holt**'s method (tendência only) are fit here, and their resíduos are tested for ruído branco (same Ljung-Box test as Step C). 
# If real structure remains in the resíduos, that's the evidence these simpler models are leaving patterns unmodeled.
# 
# - Insight principal: do Simple/Holt's resíduos still show significativo autocorrelation, confirming a sazonal term is needed?

# %%
ts_train = ts_train.asfreq('D')
ts_holdout = ts_holdout.asfreq('D')

# Simple Exponential Smoothing: no trend, no seasonality
model_simple = SimpleExpSmoothing(ts_train).fit()
resid_simple = model_simple.resid

# Holt's method: trend, no seasonality
model_holt = Holt(ts_train).fit()
resid_holt = model_holt.resid

for name, resid in [('Simple Exp. Smoothing', resid_simple), ("Holt's (trend only)", resid_holt)]:
    lb = acorr_ljungbox(resid, lags=[7, 14], return_df=True)
    print(f"\n{name}: AIC = {model_simple.aic if 'Simple' in name else model_holt.aic:.1f}")
    print(lb)


# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - **Result**: both models fail decisively. 
#     - Simple Exp. Smoothing: Ljung-Box p ≈ 1.08e-165 (lag 7); 
#     - Holt's: p ≈ 1.09e-144 (lag 7), 
#     - both far below any reasonable threshold, meaning strong autocorrelation remains in the resíduos.
# 
# - Holt's AIC (29,921.9) is actually *worse* than Simple's (29,564.3), despite adding a tendência term
#     - the added complexity doesn't pay off while weekly sazonalidade is still unmodeled and dominating the residual structure.
# 
# - **Conclusão**: neither model captures the series adequately. The residual autocorrelation is consistent with the unmodeled weekly sazonalidade (lag 7) confirmed throughout Steps A–E. This formally justifies moving directly to Holt-Winters (adds the missing sazonal term) rather than testing intermediate tendência-only models further.
# 

# %% [markdown]
# ### → Step F1: Suavização Exponencial
# 
# Given Step A's decomposition comparison (multiplicative residual stayed more stable over time than additive), `tendência='mul'` and `sazonal='mul'` are the leading candidates, but all four combinations are compared here by AIC rather than assumed.
# 

# %%
hw_candidates = {}
for trend_type in ['add', 'mul']:
    for seasonal_type in ['add', 'mul']:
        try:
            m = ExponentialSmoothing(
                ts_train, trend=trend_type, seasonal=seasonal_type,
                seasonal_periods=7, damped_trend=False
            ).fit()
            hw_candidates[f'trend={trend_type}, seasonal={seasonal_type}'] = m
            print(f"trend={trend_type}, seasonal={seasonal_type}  ->  AIC={m.aic:.1f}")
        except Exception as e:
            print(f"trend={trend_type}, seasonal={seasonal_type}  ->  failed ({e})")

# Also test damped trend on the best-performing combination once identified above

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - **Result**: AICs are close across all four combinations (28,253.1–28,264.9). The winner is **tendência=mul, sazonal=add** (AIC 28,253.1), though it's nearly tied with tendência=add, sazonal=add (28,253.7), a 0.6-point gap is not a strong signal either way.
# 
# - **Note of caution**: this contradicts the expectation from Step A's decomposition (where the multiplicative model's residual looked more stable). Here, the *sazonal* component actually prefers additive in both top candidates.
#     - the formal AIC comparison refines/complicates the earlier visual read.
# 
# - **Selected configuration**: `tendência='mul', sazonal='add', sazonal_periods=7`: will also compare with `damped_tendência=True` before finalizing.

# %% [markdown]
# #### Comparação de Tendência Amortecida
# 
# Testing whether damping the tendência (letting growth flatten out over the previsão horizon, rather than continuing indefinitely) improves the winning configuration. Relevant since Step J previsãos 30 days out, where an undamped multiplicative tendência can compound aggressively.

# %%
model_undamped = ExponentialSmoothing(
    ts_train, trend='mul', seasonal='add', seasonal_periods=7, damped_trend=False
).fit()

model_damped = ExponentialSmoothing(
    ts_train, trend='mul', seasonal='add', seasonal_periods=7, damped_trend=True
).fit()

print(f"Undamped: AIC = {model_undamped.aic:.1f}")
print(f"Damped:   AIC = {model_damped.aic:.1f}")

if hasattr(model_damped, 'params') and 'damping_trend' in model_damped.params:
    print(f"\nDamping parameter (phi): {model_damped.params['damping_trend']:.4f}")
    print("(phi close to 1.0 = minimal damping; phi well below 1.0 = meaningful flattening)")

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - **Result**: undamped AIC = 28,253.1 vs. damped AIC = 28,259.2: damping worsens fit by 6.1 points.
# 
# - **Damping parameter (phi) = 0.995**, essentially equivalent to no damping (phi=1.0 = fully undamped), the model finds almost no evidence that tendência growth is flattening within the treinamento period.
# 
# - **Decisão**: keep the **undamped** model. The extra damping parameter doesn't earn its complexity here, both by AIC and by the phi value itself showing negligible damping.
# 
# - **Caveat for Step J**: this doesn't mean growth will *literally* continue undamped forever, it means the treinamento data (through Dec/2020) doesn't show enough of a flattening signal yet for the model to justify assuming one. Worth revisiting if the 30-day March/2021 previsão (Step J) looks implausibly aggressive, since Holt-Winters tendência extrapolation can compound quickly over a previsão horizon even when phi is technically ~1.0.
# 

# %% [markdown]
# #### Residual Verificação de Ruído Branco
# 
# Same Ljung-Box + ACF approach used for SARIMA's resíduos, now applied to the final Holt-Winters model from F1 (`tendência='mul', sazonal='add', sazonal_periods=7, damped_tendência=False`).
# 
# - Insight principal: does Holt-Winters leave more or less residual structure than SARIMA did?

# %%
model_hw_final = ExponentialSmoothing(
    ts_train, trend='mul', seasonal='add', seasonal_periods=7, damped_trend=False
).fit()

residuals_hw = model_hw_final.resid

lb_resid_hw = acorr_ljungbox(residuals_hw, lags=[7, 14, 21, 30], return_df=True)
print(lb_resid_hw)

fig, ax = plt.subplots(figsize=(14, 4))
plot_acf(residuals_hw.dropna(), lags=30, ax=ax, alpha=0.05)
ax.set_title('ACF of Holt-Winters Residuals (should look like white noise)')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - **Result**: all four lags fail decisively (p ≈ 1e-17 to 1e-29), Holt-Winters resíduos show strong, unambiguous autocorrelation. Even lag 7, which SARIMA passed, fails here.
# 
# - **Interpretation**: this specific Holt-Winters configuration (tendência='mul', sazonal='add') has not fully captured the series' structure, meaningful patterns remain in the resíduos that the model isn't representing.
# 
# - **Likely cause**: Holt-Winters applies a single fixed sazonal shape and has no error/MA-type term to absorb residual autocorrelation, unlike SARIMA's `ma.L1`/`ma.S.L7` terms, which is consistent with SARIMA clearing the lag-7 test where Holt-Winters did not.
# 
# - **Takeaway**: on this white-noise criterion alone, there's more available signal being left on the table by the Holt-Winters model than by the SARIMA model, worth keeping in mind heading into Step G, though final selection should still weigh holdout accuracy (Step I) alongside this diagnostic, not this test in isolation.
# 

# %% [markdown]
# #### ▫️Modelo Final F1
# 
# **Configuração**: `ExponentialSmoothing(tendência='mul', sazonal='add', sazonal_periods=7, damped_tendência=False)`
# 
# This is F1's candidate to carry forward into Step G's model comparison against SARIMA/SARIMAX.
# 

# %% [markdown]
# ### → Step F2: ARIMA / SARIMA (identificação, estimação, variáveis exógenas, coeficientes)
# 

# %% [markdown]
# 
# #### Identificação de Parâmetros (p, q): Não Sazonal
# 
# Using `ts_seasdiff` (`ts_train.diff(7)`), the series that achieved stationarity in Step D with **d=0, D=1, s=7**. The ACF informs `q`, the PACF informs `p`.
# 
# - Insight principal: Where do the ACF/PACF cut off or decay, suggesting p and q?

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(ts_seasdiff, lags=30, ax=ax1, alpha=0.05)
ax1.set_title('ACF of D=1 differenced series (informs q)')
ax1.grid(alpha=0.3)

plot_pacf(ts_seasdiff, lags=30, ax=ax2, alpha=0.05)
ax2.set_title('PACF of D=1 differenced series (informs p)')
ax2.grid(alpha=0.3)

plt.tight_layout(); plt.show()

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - **q** (MA order): last lag with a significativo spike in the ACF before it cuts off/decays (non-sazonal lags: 1, 2, 3...).
# 
# - **p** (AR order): last lag with a significativo spike in the PACF (non-sazonal lags).
# 
# - Recall from Passo D: the D=1-only panel showed a mild lag-1 spike (~0.22 ACF) and a residual lag-7 spike (~-0.3), expect lag-1 to be the main non-sazonal candidate here.
# 
# - PACF cuts off faster (lag 2–3) than the ACF decays (out to lag 3), the classic AR signature, suggesting **p ∈ {1, 2}, q = 0** as the starting hypothesis.
# 
# - The visual read isn't perfectly clean-cut (the panels decay gradually rather than showing one sharp spike), so rather than commit to a single combination by eye, the candidates below are tested in a grid search and selected by AIC, standard practice when ACF/PACF suggest a range rather than one obvious answer.
# 

# %% [markdown]
# #### Identificação de Parâmetros (P, Q, s): Sazonal
# 
# Same ACF/PACF, now read at the sazonal lags (7, 14, 21...) rather than the early non-sazonal ones.
# 
# - Insight principal: do the sazonal lags show a single clean spike (pure AR or MA), or persist across multiple sazonal lags in both panels (mixed)?

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - **Q** (sazonal MA order): significativo spike at the sazonal lags (7, 14, 21...) in the ACF.
# 
# - **P** (sazonal AR order): significativo spike at the sazonal lags in the PACF.
# 
# - **s = 7** (weekly, confirmed since Step A).
# 
# - Significant spikes persist across lags 7, 14, and (in PACF) 21 in both ACF and PACF, a mixed sazonal pattern, suggesting **P = 1, Q = 1**, s = 7 as the starting hypothesis.
# 
# - As with the non-sazonal identification, this visual read is a starting range rather than a final answer, confirmed (or revised) by the grid search below.
# 

# %% [markdown]
# **Candidates to test**:
# - (p, d, q)
#     - p ∈ {0, 1, 2}
#     - d = 0
#     - q ∈ {0, 1}
# 
# - (P, D, Q)s
#     - P ∈ {0, 1}
#     - D = 1
#     - Q ∈ {0, 1}
#     - s = 7

# %% [markdown]
# #### Busca em Grade: Seleção de (p,q)(P,Q) por AIC
# 
# Testing the candidate range identified from the ACF/PACF (p ∈ {0,1,2}, q ∈ {0,1}, P ∈ {0,1}, Q ∈ {0,1}, with d=0, D=1, s=7 fixed from Step D) and ranking by AIC/BIC.
# 
# - Insight principal: which combination wins, and does it match the p=1-2, q=0, P=1, Q=1 hypothesis from the visual read?

# %%
d, D, s = 0, 1, 7
p_range = range(0, 3)
q_range = range(0, 2)
P_range = range(0, 2)
Q_range = range(0, 2)

results = []
warnings.filterwarnings('ignore')

for p, q, P, Q in itertools.product(p_range, q_range, P_range, Q_range):
    try:
        model = sm.tsa.statespace.SARIMAX(
            ts_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        results.append({
            'order': (p, d, q), 'seasonal_order': (P, D, Q, s),
            'AIC': model.aic, 'BIC': model.bic
        })
    except Exception as e:
        continue

warnings.filterwarnings('default')

results_df = pd.DataFrame(results).sort_values('AIC').reset_index(drop=True)
# print(results_df.head(20))

# %%
print(results_df.sort_values('AIC', ascending=True))

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - Full grid confirms the earlier result: **(1,0,1)(1,1,1,7)** wins on both AIC (30,919.0) and BIC (30,943.9).
# 
# - Pattern across the tabela: the sazonal MA term (Q=1) drives the largest AIC improvements, e.g., (1,0,1) alone improves from 31,382.9 (Q=0,P=0) to 30,919.0 (P=1,Q=1), a ~460-point gain. This confirms weekly sazonalidade (already established in Steps A/D) is the dominant structural signal the model needs to capture, more so than fine-tuning the non-sazonal AR/MA terms.
# 
# - The weakest model, (0,0,0)(0,1,0,7) (differencing alone, no ARMA terms) ranks last (AIC 31,480.7), confirming that ARMA structure adds real explanatory value beyond simple sazonal differencing.
# 
# - **Final selected parameters**: p=1, d=0, q=1, P=1, D=1, Q=1, s=7
# 

# %% [markdown]
# #### Comparação da Família de Modelos: ARMA → ARIMA → SARIMA → SARIMAX
# 
# Rather than jumping straight to the final SARIMA/SARIMAX, this section fits the intermediate model families using the same (p,q)=(1,1) identified above, progressively adding what each earlier step showed was missing:
# 
# - **ARMA(1,1)**: no differencing at all, included as a baseline, despite Step D confirming the raw series is non-stationary.
# - **ARIMA(1,1,1)**: regular differencing only (d=1), Step D showed this removes tendência but leaves weekly sazonalidade (lags 7/14/21) fully intact.
# - **SARIMA(1,0,1)(1,1,1,7)**: adds sazonal differencing/terms, the model selected via the grid search above.
# - **SARIMAX**: SARIMA + `Pagamento`, `Feriado`, `Seasonal_Index`, the final model from this section.
# 
# Each is evaluated on both treinamento fit (AIC/BIC) and holdout accuracy (MAE/RMSE/MAPE).
# 
# - Insight principal: does each added layer of complexity improve holdout accuracy, or does it plateau/reverse, echoing the F1-vs-F2 result from Step G?

# %%
model_family_results = []

# Local setup: holdout actuals, excluding isClosed days
isclosed_holdout = df_complete.set_index('Date').loc[ts_holdout.index, 'isClosed']
holdout_mask = isclosed_holdout == 0
actual_holdout = ts_holdout[holdout_mask]

# %% [markdown]
# ##### Estimação do Modelo: ARMA

# %%
# 1. ARMA(1,1) -- no differencing
model_arma = sm.tsa.statespace.SARIMAX(
    ts_train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False, enforce_invertibility=False
).fit(disp=False)
forecast_arma = model_arma.get_forecast(steps=len(ts_holdout)).predicted_mean
metrics_arma = evaluate_forecast(actual_holdout, forecast_arma.loc[actual_holdout.index], 'ARMA(1,1)')
model_family_results.append({**metrics_arma, 'AIC': model_arma.aic, 'BIC': model_arma.bic})

print(model_arma.summary())

# %% [markdown]
# ##### Estimação do Modelo: ARIMA

# %%
# 2. ARIMA(1,1,1) -- regular differencing, no seasonal terms
model_arima = sm.tsa.statespace.SARIMAX(
    ts_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False, enforce_invertibility=False
).fit(disp=False)
forecast_arima = model_arima.get_forecast(steps=len(ts_holdout)).predicted_mean
metrics_arima = evaluate_forecast(actual_holdout, forecast_arima.loc[actual_holdout.index], 'ARIMA(1,1,1)')
model_family_results.append({**metrics_arima, 'AIC': model_arima.aic, 'BIC': model_arima.bic})

print(model_arima.summary())

# %%
family_comparison_df = pd.DataFrame(model_family_results)[['Model','AIC','BIC','MAE','RMSE','MAPE']]
print(family_comparison_df.to_string(index=False))

# %% [markdown]
# ##### Estimação do Modelo: SARIMA
# 
# Fitting the selected model **(1,0,1)(1,1,1,7)** on `ts_train`, then checking whether its resíduos behave like ruído branco (same Ljung-Box + ACF approach as Step C) before considering exogenous variables.
# 
# - Insight principal: do the resíduos pass the ruído branco check, confirming the model has captured the available structure?

# %%
model_sarima = sm.tsa.statespace.SARIMAX(
    ts_train,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

forecast_sarima = model_sarima.get_forecast(steps=len(ts_holdout)).predicted_mean
metrics_sarima = evaluate_forecast(actual_holdout, forecast_sarima.loc[actual_holdout.index], 'SARIMA(1,0,1)(1,1,1,7)')
model_family_results.append({**metrics_sarima, 'AIC': model_sarima.aic, 'BIC': model_sarima.bic})

print(model_sarima.summary())

# %% [markdown]
# ###### Residual Verificação de Ruído Branco
# 
# Same test as Step C, now applied to the model's resíduos instead of the raw series, this time we *want* to fail to reject H0 (resíduos = ruído branco = no structure left to capture).
# 
# - H0: resíduos are ruído branco (model has captured all available structure). 
# - H1: resíduos still have structure (model is missing something).
# - Decisão rule: p-value > 0.05 across the tested lags → good, no significativo leftover structure.

# %%
residuals_sarima = model_sarima.resid

lb_resid = acorr_ljungbox(residuals_sarima, lags=[7, 14, 21, 30], return_df=True)
print(lb_resid)

fig, ax = plt.subplots(figsize=(14, 4))
plot_acf(residuals_sarima.dropna(), lags=30, ax=ax, alpha=0.05)
ax.set_title('ACF of SARIMA Residuals — should look like white noise')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ###### ▪️<span style="color:purple">Notas</span>
# 
# - **Ljung-Box on resíduos**: passes at lag 7 (p=0.108) but fails at lags 14, 21, 30 (p=0.014, 0.024, 0.000013), the model captures the immediate weekly pattern but leaves longer-range structure unexplained.
# 
# - **Answering the ruído branco question directly**: yes, there are still patterns to uncover, SARIMA has captured the dominant weekly (lag-7) signal, but something beyond a single week is still leaking into the resíduos.
# 
# - **Hypothesis for the remaining structure**: `Pagamento` and `Feriado` don't follow a clean 7-day rhythm (paydays land on a specific business day of the month; holidays are calendar-fixed), a pure sazonal ARMA structure has no way to represent that kind of event-driven timing. These are natural candidates for explaining the lag-14/21/30 autocorrelation that remains.
# 
# - **Next step**: fit SARIMAX with `Pagamento`, `Vale` and `Feriado` as exogenous regressors (even `the sazonalidade index created by month) and re-run this same Ljung-Box test, if the lag-14+ p-values move above 0.05, that's direct evidence the exogenous variables are absorbing the leftover structure rather than being added speculatively.
# 

# %% [markdown]
# ##### Estimação do Modelo: SARIMAX (SARIMA with Exogenous Variables)
# 
# Testing whether `Pagamento`, `Feriado`, `Vale`, `isClosed` and `Seasonal_Index` explain the lag-14/21/30 residual structure the base SARIMA left uncaptured. 
# 
# - Insight principal: do the exogenous coefficients come out significativo, and does the Ljung-Box test now pass at lags 14/21/30?

# %%
exog_cols = ['Pagamento', 'Feriado', 'Vale', 'isClosed', 'Seasonal_Index']

exog_train = df_complete.set_index('Date').loc[ts_train.index, exog_cols].asfreq('D')
exog_holdout_forecast = df_complete.set_index('Date').loc[ts_holdout.index, exog_cols].asfreq('D')

model_sarimax_1 = sm.tsa.statespace.SARIMAX(
    ts_train,
    exog=exog_train,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

forecast_sarimax = model_sarimax_1.get_forecast(steps=len(ts_holdout), exog=exog_holdout_forecast).predicted_mean
metrics_sarimax = evaluate_forecast(actual_holdout, forecast_sarimax.loc[actual_holdout.index], 'SARIMAX(1,0,1)(1,1,1,7) X: Pagamento/Feriado/Vale/isClosed/Seasonal_Index')
model_family_results.append({**metrics_sarimax, 'AIC': model_sarimax_1.aic, 'BIC': model_sarimax_1.bic})

print(model_sarimax_1.summary())

# %% [markdown]
# ###### ▪️<span style="color:purple">Notas</span>
# 
# **Model included 5 exogenous variables** (`Pagamento`, `Feriado`, `Vale`, `isClosed`, `Seasonal_Index`) findings below cover all five as run.
# 
# - **Pagamento**: coef +1.97e5, p=0.024
#     - **significativo**, consistent with Part 1's Mann-Whitney result (p=0.0014).
# 
# - **Feriado**: coef -3.20e5, p<0.001
#     - now **clearly significativo**, notably stronger than Part 1's Mann-Whitney result (p=0.0645, borderline). 
#     - Once tendência/sazonalidade/other exogenous effects are controlled for, which SARIMAX does and a simple group-mean comparison doesn't, the holiday effect comes through much more clearly. 
#     - Resolves the earlier open question about whether Feriado's raw effect was reliable.
# 
# - **Vale**: coef -6.10e4, p=0.277
#     - **not significativo**, consistent with Part 1's Mann-Whitney result (p=0.52). 
#     - Candidate to drop from a refined model.
# 
# - **isClosed**: coef -2.25e6, p<0.001
#     - **hugely significativo**, but near-tautological (sales ≈ 0 on those 7 days by construction, same relationship flagged in Part 1's correlation check, r=-0.220). 
#     - **Decisão needed**: let the model *learn* this coefficient, or *enforce* previsão≈0 on isClosed dates as a hard rule in Step J instead.
# 
# - **Seasonal_Index**: coef +1.48e6, p<0.001
#     - **strongly significativo**, confirming the Nov/Dec effect from Step E adds real explanatory power beyond the base sazonal ARMA structure.
# 
# 
# **Fit improvement**: AIC 30,569.5 / BIC 30,619.3, vs. base SARIMA's 30,919.0 / 30,943.9, a ~350-point AIC improvement, easily earning the added complexity.
# 
# 
# **Diagnostics improved alongside fit**: numerical stability condition number improved (6.26e25 vs. base model's 1.06e39, though still flagged), and non-normality eased (JB=2,382, kurtosis=10.1 vs. base model's 14,647/20.9), the exogenous variables are absorbing some of what was producing the most extreme resíduos.
# 
# 
# **Still open**: the built-in `Prob(Q)=0.83` in the summary is only the lag-1 diagnostic, the fuller Ljung-Box test (lags 7/14/21/30) from the dedicated residual-check cell hasn't been run on this SARIMAX model yet. That's the test that will confirm whether the exogenous variables actually resolved the lag-14+ structure the base SARIMA left uncaptured.
# 
# 
# **Next steps**:
# 1. run the full Ljung-Box check on `model_sarimax.resid`, 
# 2. decide whether to drop `Vale` given its non-significance, 
#     - Let's drop `Vale`.
# 3. decide how `isClosed` should be handled going into Step J.
#     - Better to enforce 0 everytime these days are in the prediction horizon.
# 

# %% [markdown]
# #### Reajustando SARIMAX: Conjunto Final de Exógenas
# 
# Based on the decisions above: **`Vale` dropped** (not significativo, p=0.277). 
# **`isClosed` dropped from the fitted exogenous set**, rather than let the model estimate its effect, it will be enforced as a hard rule in Step J (force previsão ≈ 0 whenever a future date falls on Dec-25 or Jan-01), which is more robust than relying on a learned coefficient for a near-deterministic outcome.
# 
# Final exogenous set: **`Pagamento`, `Feriado`, `Seasonal_Index`**.

# %%
exog_cols_final = ['Pagamento', 'Feriado', 'Seasonal_Index']

exog_train_final = df_complete.set_index('Date').loc[ts_train.index, exog_cols_final].asfreq('D')
exog_holdout_forecast_final = df_complete.set_index('Date').loc[ts_holdout.index, exog_cols_final].asfreq('D')

model_sarimax = sm.tsa.statespace.SARIMAX(
    ts_train,
    exog=exog_train_final,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

forecast_sarimax = model_sarimax.get_forecast(steps=len(ts_holdout), exog=exog_holdout_forecast_final).predicted_mean
metrics_sarimax = evaluate_forecast(actual_holdout, forecast_sarimax.loc[actual_holdout.index], 'SARIMAX(1,0,1)(1,1,1,7) X: Pagamento/Feriado/Seasonal_Index')
model_family_results.append({**metrics_sarimax, 'AIC': model_sarimax.aic, 'BIC': model_sarimax.bic})

print(model_sarimax.summary())

# %%
family_comparison_df = pd.DataFrame(model_family_results)[['Model','AIC','BIC','MAE','RMSE','MAPE']]
print(family_comparison_df.to_string(index=False))

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - **Pagamento**: p rose to 0.090 (not significativo at α=0.05), up from p=0.024 in the 5-variable version and p=0.0014 in Part 1's Mann-Whitney test. Coefficient itself barely changed (1.981e5 vs. 1.973e5), the shift is driven by a wider standard error (1.17e5 vs. 8.74e4), a side effect of removing `isClosed`, which leaves more unexplained variance in the resíduos overall. Not evidence the effect isn't real, a consequence of the isClosed decision.
# 
# - **Feriado**: remains clearly significativo (p<0.001), coefficient stable (-3.16e5 vs. -3.20e5).
# 
# - **Seasonal_Index**: remains clearly significativo (p<0.001), coefficient stable (+1.50e6 vs. +1.48e6).
# 
# - **Ajuste**: AIC 30.832,1 / BIC 30.872,0 — pior que a versão de 5 variáveis (30.569,5/30.619,3) em ~262 pontos, mas ainda ~87 pontos melhor que o SARIMA base (30.919,0). O conjunto mais enxuto de 3 variáveis ainda adiciona valor real, apenas menos do que o conjunto completo de 5 variáveis.
# 
# - **Diagnostics**: JB/curtose reverteram para níveis próximos ao modelo base (13.804/20,5 vs. base 14.647/20,9), confirmando que `isClosed` estava absorvendo grande parte do comportamento de resíduos extremos. Este é um trade-off aceito ao mover `isClosed` para uma regra rígida no Passo J em vez de um coeficiente ajustado.
# 
# - **Still open**: o Ljung-Box completo (defasagens 7/14/21/30) ainda não foi rodado neste modelo final — esse é o teste que confirma se Pagamento/Feriado/Seasonal_Index resolveram a estrutura das defasagens 14+ por conta própria, sem a ajuda de isClosed.
# 
# 

# %% [markdown]
# #### Residual Verificação de Ruído Branco
# 
# - H0: resíduos are ruído branco (model has captured all available structure). 
# - H1: resíduos still have structure (model is missing something).
# - Decisão rule: p-value > 0.05 across the tested lags → good, no significativo leftover structure.

# %%
residuals_sarimax = model_sarimax.resid

lb_resid_sarimax = acorr_ljungbox(residuals_sarimax, lags=[7, 14, 21, 30], return_df=True)
print(lb_resid_sarimax)

fig, ax = plt.subplots(figsize=(14, 4))
plot_acf(residuals_sarimax.dropna(), lags=30, ax=ax, alpha=0.05)
ax.set_title('ACF of Final SARIMAX Residuals')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - **Result**: Defasagem 7 passa (p=0,113, consistente com SARIMA base). Defasagem 21 agora passa (p=0,128, melhorou do p=0,024 que falhava no SARIMA base) — uma vitória genuína. Defasagens 14 e 30 ainda falham (p=0,030, p=0,003), embora ambas tenham melhorado substancialmente em relação ao modelo base (p=0,014→0,030, p=0,000013→0,003).
# 
# - **Conclusão**: `Pagamento`, `Feriado` e `Seasonal_Index` explicam parte, mas não toda, a estrutura residual que o SARIMA base deixou para trás. A hipótese é parcialmente confirmada — variáveis exógenas ajudam, mas algum padrão permanece, particularmente no horizonte mensal (defasagem ~30).
# 
# - **Plausible remaining source**: `Vale` (17th business day, a monthly-not-weekly event) was deliberately excluded for non-significance in isolation, but it's the one dropped variable tied to a non-weekly rhythm that could still explain some of the lag-21/28-adjacent structure. Worth a documented limitation rather than re-adding it purely to force a pass, given its own earlier test was clearly not significativo on its own.
# 
# - **Decisão**: dado que a melhoria é real mas incompleta, este é um ponto razoável para aceitar o SARIMAX atual de 3 variáveis como modelo de trabalho para a comparação do Passo G, notando explicitamente a limitação residual (alguma estrutura de não-ruído-branco permanece, principalmente perto do horizonte mensal) como parte dos diagnósticos do Passo H, em vez de tratá-la como totalmente resolvida.
# 

# %% [markdown]
# #### Testando `Vale` como 4ª Variável Exógena
# 
# Readicionando `Vale` (mensal, 17º dia útil — a única variável removida ligada a um ritmo não semanal) para verificar se resolve a estrutura residual das defasagens 14/30, mantendo `isClosed` excluído conforme a decisão de regra rígida do Passo J.
# 
# - Insight principal: does adding Vale back close the gap at lags 14/30, and is its coefficient significativo this time in a leaner model?

# %%
exog_cols_v2 = ['Pagamento', 'Feriado', 'Seasonal_Index', 'Vale']

exog_train_v2 = df_complete.set_index('Date').loc[ts_train.index, exog_cols_v2].asfreq('D')
exog_holdout_v2 = df_complete.set_index('Date').loc[ts_holdout.index, exog_cols_v2].asfreq('D')

model_sarimax_v2 = sm.tsa.statespace.SARIMAX(
    ts_train,
    exog=exog_train_v2,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

forecast_sarimax_v2 = model_sarimax_v2.get_forecast(steps=len(ts_holdout), exog=exog_holdout_v2).predicted_mean
metrics_sarimax_v2 = evaluate_forecast(actual_holdout, forecast_sarimax_v2.loc[actual_holdout.index], 'SARIMAX(1,0,1)(1,1,1,7) X: Pagamento/Feriado/Seasonal_Index/Vale')
model_family_results.append({**metrics_sarimax_v2, 'AIC': model_sarimax_v2.aic, 'BIC': model_sarimax_v2.bic})

print(model_sarimax_v2.summary())

# %%
residuals_sarimax_v2 = model_sarimax_v2.resid

lb_resid_v2 = acorr_ljungbox(residuals_sarimax_v2, lags=[7, 14, 21, 30], return_df=True)
print(lb_resid_v2)

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - **Vale**: p=0.235, not significativo, the third independent test to reach this conclusion (Mann-Whitney p=0.52 in Part 1; 5-variable SARIMAX p=0.277; this 4-variable SARIMAX p=0.235). Consistent evidence across three different tests that Vale has no meaningful relationship with sales at this level of aggregation.
# 
# - **Ajuste**: AIC 30.836,9 vs. 30.832,1 do modelo de 3 variáveis — Vale custa 4,8 pontos de AIC em vez de melhorar o ajuste; o parâmetro adicionado não justifica sua complexidade.
# 
# - **Ljung-Box**: essencialmente inalterado em relação ao modelo de 3 variáveis (defasagem 14: 0,030→0,034, defasagem 21: 0,128→0,138, defasagem 30: 0,003→0,005) — diferenças negligenciáveis, confirmando que Vale não é a fonte da estrutura residual restante nas defasagens 14/30.
# 
# - **Decisão**: reject `Vale` as a regressor. **Final SARIMAX model: `Pagamento`, `Feriado`, `Seasonal_Index`** (order=(1,0,1), sazonal_order=(1,1,1,7)).
# 
# - **Documented limitation carried into Step H**: some residual structure remains at lags 14 and 30 that this model doesn't explain. Tested candidates (Vale) have been ruled out as the cause; the remaining structure is left as an acknowledged limitation rather than chased further, since continuing to add untested variables risks overfitting to the treinamento data without a clear business hypothesis behind them.
# 

# %% [markdown]
# #### Comparação

# %%
family_comparison_df = pd.DataFrame(model_family_results)[['Model','AIC','BIC','MAE','RMSE','MAPE']]
print(family_comparison_df.to_string(index=False))

# %%
print(family_comparison_df.sort_values('AIC', ascending=True).to_string(index=False))

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# **Comparação completa da família de modelos** (incluindo o SARIMAX de 5 variáveis reconsiderado):
# 
# | Model | AIC | BIC | MAE | RMSE | MAPE |
# |---|---|---|---|---|---|
# | SARIMAX + Pagamento/Feriado/Vale/isClosed/Seasonal_Index | **30,569.5** | **30,619.3** | **420,206** | **503,946** | **16.5%** |
# | SARIMAX + Pagamento/Feriado/Seasonal_Index | 30,832.1 | 30,872.0 | 492,505 | 566,727 | 19.0% |
# | SARIMAX + Pagamento/Feriado/Seasonal_Index/Vale | 30,836.9 | 30,881.8 | 497,278 | 572,441 | 19.2% |
# | SARIMA(1,0,1)(1,1,1,7) | 30,919.0 | 30,943.9 | 548,190 | 679,328 | 21.8% |
# | ARIMA(1,1,1) | 32,602.3 | 32,617.3 | 754,761 | 868,400 | 30.8% |
# | ARMA(1,1) | 32,655.4 | 32,670.4 | 817,224 | 917,547 | 33.8% |
# 
# - **Reabrindo a decisão sobre isClosed**: the 5-variable model wins on every metric, including holdout MAE/RMSE/MAPE evaluated *only on non-isClosed days*. This means including `isClosed` isn't just improving predictions on the 7 closed days themselves, it's improving the model's fit on ordinary days too, by preventing those extreme outliers from distorting the estimation of the AR/MA/sazonal coefficients (consistent with Step H's finding that the 5-variable model's residual distribution was far better-behaved: JB=2,382 vs. 13,804 for the 3-variable version).
# 
# - **Recomendação revisada**: use the 5-variable SARIMAX (with `isClosed` as a *fitted* regressor) as the final model, rather than the 3-variable version, the accuracy benefit is real and applies broadly, not just to closed days. The Step J hard-override plan can still be layered on top as a safety net (force previsão to ≈0 specifically on isClosed dates, regardless of what the model predicts), combining the best of both: a cleaner-fitting model *and* a guaranteed-correct output on the 7 fully deterministic dates.
# 
# - Isso também significa que a comparação SARIMAX-vs-Holt-Winters do Passo G provavelmente deve ser revisitada usando este SARIMAX melhorado de 5 variáveis em vez da versão de 3 variáveis que perdeu para Holt-Winters — a perda anterior (MAE 492.505 vs. 436.193 de Holt-Winters) pode não se manter com este modelo melhor especificado.
# 
# 

# %%
lb_resid_final = acorr_ljungbox(model_sarimax.resid, lags=[7, 14, 21, 30], return_df=True)
print(lb_resid_final)

# %% [markdown]
# - Teste formal de Ljung-Box nos resíduos do modelo SARIMAX final (com isClosed reintegrado):
# 
# | Lag | 3 variáveis (anterior) | 4 variáveis (final) |
# |---|---|---|
# | 7 | p=0.113 (passa) | p=0.139 (passa) |
# | 14 | p=0.030 (falha) | p=0.0003 (falha) |
# | 21 | p=0.128 (passa) | p=0.0003 (falha) |
# | 30 | p=0.003 (falha) | p<0.001 (falha) |
# 
# - **Conclusão**: reintegrar `isClosed` melhorou AIC/BIC e acurácia do holdout substancialmente (Passo F2/G), mas **não** melhorou a brancura residual — a defasagem 21 na verdade passou de aprovação para falha, e as defasagens 14/30 pioraram marcadamente. Isso significa que acurácia e diagnósticos de resíduos estão respondendo perguntas diferentes aqui e não se movem juntos — a mesma lição já vista na comparação F1-vs-F2 (Passo G), agora aparecendo novamente dentro das próprias escolhas de variáveis de F2.
# 
# - **Interpretation**: o efeito grande e limpo de `isClosed` absorve variância substancial que antes ficava no termo de erro. Isso provavelmente torna a variância residual *restante*, menor, mais sensível a qualquer estrutura que ainda esteja lá — o mesmo padrão residual absoluto se torna estatisticamente mais fácil de detectar (ou os termos ARMA do modelo se deslocaram no ajuste para compensar a inclusão de isClosed, mudando a forma do padrão residual).
# 
# - **Decisão**: this is documented as a known, accepted limitation. Holdout accuracy (the metric that most directly reflects genuine previsãoing performance and the business goal) was prioritized over residual whiteness when the two disagreed, consistent with the same reasoning applied earlier when selecting SARIMAX over Holt-Winters despite AIC favoring SARIMAX and holdout favoring the opposite in that case.

# %% [markdown]
# ### ▫️Modelo Final F2
# 
# **Configuração**: `SARIMAX(order=(1,0,1), sazonal_order=(1,1,1,7), exog=['Pagamento', 'Feriado', 'Seasonal_Index', 'isClosed'])`
# 
# **Caminho de seleção**:
# - SARIMA base (1,0,1)(1,1,1,7) escolhido via busca em grade de 24 combinações AIC/BIC (identificação do Passo F2), vencedor em ambas as métricas.
# 
# - Base model's resíduos passed ruído branco at lag 7 but failed at lags 14/21/30 (Ljung-Box), motivated testing exogenous regressors.
# 
# - 5-variable SARIMAX (`Pagamento`, `Feriado`, `Vale`, `isClosed`, `Seasonal_Index`) tested first: best fit (AIC 30,569.5). `Vale` was dropped (non-significativo across three independent tests: Mann-Whitney, and two separate SARIMAX specifications).
# 
# - `isClosed` was initially moved out of the fitted exogenous set on the reasoning that its effect is near-tautological (sales ≈0 on those 7 days by construction) and better handled deterministically via a Step J hard rule. However, the full model-family comparison (ARMA → ARIMA → SARIMA → SARIMAX) showed this decision came at a real accuracy cost: **the model including `isClosed` outperformed the model without it on every metric, AIC, BIC, and holdout MAE/RMSE/MAPE, even when holdout accuracy was measured only on non-isClosed days.** This indicates `isClosed` isn't just improving predictions on the 7 closed days themselves; leaving those extreme outliers unmodeled distorts the estimation of the AR/MA/sazonal coefficients, degrading fit on ordinary days too (consistent with the 5-variable model's cleaner residual distribution found in Passo H: JB=2,382 vs. 13,804 without isClosed).
# 
# - **Revised decision**: `isClosed` is reinstated as a fitted exogenous regressor. The Step J hard-override plan is kept as a complementary safety net on top, previsão forced to ≈0 specifically on isClosed dates, regardless of what the model predicts, combining a cleaner-fitting model with a guaranteed-correct output on the 7 fully deterministic dates.
# 
# **Coeficientes finais** (da especificação de 5 variáveis):
# - `Feriado`, `Seasonal_Index`, `isClosed`: all significativo, p<0.001
# - `Pagamento`: p=0.024, significativo
# - ARMA terms (ar.L1, ma.L1, ar.S.L7, ma.S.L7): all significativo, p<0.001
# 
# **Ajuste**: AIC 30.569,5 / BIC 30.619,3 — uma melhoria de ~350 pontos de AIC sobre o SARIMA base (30.919,0), a maior melhoria de qualquer configuração exógena testada.
# 
# Este é o candidato de F2 a levar adiante para a comparação de modelos do Passo G contra o modelo Holt-Winters de F1 — note que o resultado anterior SARIMAX-vs-Holt-Winters do Passo G (Holt-Winters vencendo) foi baseado na versão de 3 variáveis e deve ser reexecutado com esta especificação melhorada antes que a seleção final do modelo se mantenha.
# 

# %% [markdown]
# ## 🔸Passo G: Escolhendo o Melhor Modelo
# 
# **Objective**: comparar os dois candidatos:
# - Holt-Winters (F1: tendência='mul', sazonal='add') and 
# - SARIMAX (F2: (1,0,1)(1,1,1,7) + Pagamento/Feriado/Seasonal_Index/isClosed) 
# na acurácia fora da amostra contra `ts_holdout`. 
# This is the first evaluation that reflects genuine previsãoing performance rather than in-sample fit (AIC/BIC) or residual diagnostics, both of which only describe how well each model explains data it already saw.
# 
# - Insight principal: qual modelo vence na acurácia do holdout, e concorda com o que AIC/Ljung-Box sugeriram durante F2?

# %%
exog_cols_final = ['Pagamento', 'Feriado', 'Seasonal_Index', 'isClosed']
exog_train = df_complete.set_index('Date').loc[ts_train.index, exog_cols_final].asfreq('D')
exog_holdout = df_complete.set_index('Date').loc[ts_holdout.index, exog_cols_final].asfreq('D')

model_sarimax = sm.tsa.statespace.SARIMAX(
    ts_train,
    exog=exog_train,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

hw_forecast = model_hw_final.forecast(len(ts_holdout))
sarimax_forecast = model_sarimax.get_forecast(steps=len(ts_holdout), exog=exog_holdout).predicted_mean

isclosed_holdout = df_complete.set_index('Date').loc[ts_holdout.index, 'isClosed']
mask = isclosed_holdout == 0

results = []
results.append(evaluate_forecast(ts_holdout[mask], hw_forecast[mask], 'Holt-Winters (F1)'))
results.append(evaluate_forecast(ts_holdout[mask], sarimax_forecast[mask], 'SARIMAX (F2, 4-var)'))

comparison_df = pd.DataFrame(results).sort_values('RMSE')
print(comparison_df)

# %% [markdown]
# ### Visual Comparação: Actual vs. Forecast

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_train.index, ts_train.values, label='Train', color='gray', alpha=0.6)
ax.plot(ts_holdout.index, ts_holdout.values, label='Actual (holdout)', color='black', linewidth=1.5)
ax.plot(ts_holdout.index, hw_forecast.values, label='Holt-Winters forecast', color='orange')
ax.plot(ts_holdout.index, sarimax_forecast.values, label='SARIMAX forecast', color='green')
ax.legend(); ax.set_title('Holdout: Actual vs. Both Models'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_train.index[-90:], ts_train.values[-90:], label='Train (last 90 days)', color='gray', alpha=0.6)
ax.plot(ts_holdout.index, ts_holdout.values, label='Actual (holdout)', color='black', linewidth=1.5)
ax.plot(ts_holdout.index, hw_forecast.values, label='Holt-Winters forecast', color='orange')
ax.plot(ts_holdout.index, sarimax_forecast.values, label='SARIMAX forecast', color='green')
ax.legend(); ax.set_title('Holdout: Actual vs. Both Models'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Business Decisão Matrix
# 
# | Metric | Holt-Winters (F1) | SARIMAX (F2, 4-var) |
# |---|---|---|
# | MAE | 436,193 | 416,339 |
# | RMSE | 522,691 | 497,847 |
# | MAPE | 17.9% | 16.3% |
# 
# **SARIMAX vence nas três métricas de acurácia do holdout**.
# 
# 
# **Selected Model**: **SARIMAX (F2, 4-var)**, wins on the criterion that matters most for the business goal (previsão accuracy on genuinely unseen data).
# 

# %% [markdown]
# ## 🔸Passo H: Diagnóstico de Resíduos
# 
# **Objective**: examine the selected model's resíduos in full, not just the Ljung-Box test already run, to understand *how* the model is wrong, not just *whether* it captures all available structure.
# 
# - Insight principal: is the residual pattern random-looking despite failing Ljung-Box, or does it show an obvious visual pattern (e.g., still tracking the weekly ciclo)?

# %%
best_model = model_sarimax 
residuals = best_model.resid

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(residuals.index, residuals.values)
ax.axhline(0, color='red', linestyle='--')
ax.set_title('Residuals over Time (Selected Model)')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Análise da Distribuição dos Resíduos

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(residuals, kde=True, ax=axes[0])
axes[0].set_title('Residual Distribution')
sm.qqplot(residuals, line='s', ax=axes[1])
axes[1].set_title('Q-Q Plot (Normality Check)')
plt.tight_layout(); plt.show()

print(residuals.describe())

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - The histogram is roughly bell-shaped near the center, but the Q-Q plot shows clear departure from the reference line in the lower-left tail, a handful of large negative resíduos stand out from an otherwise fairly well-behaved distribution.
# 
# - **Conclusão**: não visualmente normal — melhor rodar formalmente um teste (Shapiro-Wilk, Jarque-Bera).
# 

# %% [markdown]
# ### Teste Formal de Normalidade
# 
# O gráfico Q-Q sugeriu desvio da normalidade (cauda esquerda pesada). Confirmando isso formalmente com o teste de Shapiro-Wilk, em vez de confiar apenas no julgamento visual.
# 
# - H0: resíduos are normally distributed. 
# - H1: eles não são.
# - Decisão rule: p-valor < 0,05 → rejeitar H0 → não normal.
# 
# - Insight principal: o teste confirma não normalidade, consistente com o desvio visível da cauda no gráfico Q-Q?

# %%
# Shapiro-Wilk (good general-purpose normality test)
stat_sw, p_sw = shapiro(residuals)
print(f"Shapiro-Wilk: statistic={stat_sw:.4f}, p-value={p_sw:.6f}")

# p-value validation
if p_sw < 0.05:
    print("Shapiro-Wilk: p-valor < 0.05. Residuals are NOT normally distributed.\n")
else:
    print("Shapiro-Wilk: p-valor >= 0.05. Residuals are normally distributed.\n")


# Jarque-Bera (based on skewness + kurtosis, common in time series contexts)
stat_jb, p_jb = jarque_bera(residuals)
print(f"Jarque-Bera:  statistic={stat_jb:.4f}, p-value={p_jb:.6f}")

# p-value validation
if p_jb < 0.05:
    print("Jarque-Bera: p-valor < 0.05. Residuals are NOT normally distributed.\n")
else:
    print("Jarque-Bera: p-valor >= 0.05. Residuals are normally distributed.\n")


print(f"Skewness: {residuals.skew():.4f}")
print(f"Kurtosis: {residuals.kurtosis():.4f}")


# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# 
# - **Result**: ambos os testes rejeitam a normalidade de forma decisiva. Shapiro-Wilk: estatística=0,8835 p<0,000001. Jarque-Bera: estatística=2022,9523, p<0,000001.
# 
# - **Skewness = 0.7534** (mild positive/right skew), note this corrects the earlier visual expectation of negative skew; the bulk of resíduos lean slightly positive, even though the most extreme individual outliers are negative (isClosed/holiday days).
# 
# - **Curtose = 6,5185** — este é o problema dominante, não a assimetria. Muito acima do ~0 esperado sob normalidade (convenção de excesso de curtose), indicando um centro agudamente pontiagudo com caudas pesadas — a maioria dos dias tem erros pequenos e bem comportados, mas um punhado de outliers severos ocorre em ambas as direções.
# 
# - **Practical implication**: confirms the earlier concern about prediction intervals, SARIMAX' default confidence intervals assume normally distributed errors, and with this level of excess kurtosis, those intervals will understate the true probability of an extreme miss. Worth flagging explicitly when presenting Step J's previsão intervals, and worth emphasizing RMSE (sensitive to large errors) over MAE alone when reporting Step I's accuracy.
# 

# %% [markdown]
# ### Residual Autocorrelation Verificar
# 
# Revisitando o ACF (já testado via Ljung-Box em F1) como complemento visual — onde especificamente a estrutura residual está concentrada?

# %%
fig, ax = plt.subplots(figsize=(14, 4))
plot_acf(residuals.dropna(), lags=30, ax=ax, alpha=0.05)
ax.set_title('Residual ACF — Holt-Winters')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - O gráfico ACF mostra várias barras ultrapassando a banda de confiança de 95%, incluindo na defasagem semanal (7), confirmando visualmente o que o teste de Ljung-Box de F1 já estabeleceu numericamente (p≈0 em todas as defasagens testadas).
# 
# - **Conclusão**: residual autocorrelation is real and not fully resolved, the sazonal component isn't completely absorbing the weekly rhythm, meaning some predictable pattern is still being left on the table by this model.
# 

# %%
fig, ax = plt.subplots(figsize=(14, 5))
# ax.plot(ts_train.index[-1:], ts_train.values[-1:], label='Train (last 1 days)', color='gray', alpha=0.6)
ax.plot(ts_holdout.index, ts_holdout.values, label='Actual (holdout)', color='black', linewidth=1.5)
ax.plot(ts_holdout.index, hw_forecast.values, label='Holt-Winters forecast', color='orange')
ax.plot(ts_holdout.index, sarimax_forecast.values, label='SARIMAX forecast', color='green')
ax.legend(); ax.set_title('Holdout: Actual vs. Both Models'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Quantificando o Viés de Pico/Vale
# 
# A leitura visual sugere erro sistemático, não aleatório: Holt-Winters superprevê os vales; SARIMAX subprevê tanto picos quanto vales. Dividir o holdout em "dias de pico" (top 25% das vendas reais) e "dias de vale" (bottom 25%) e verificar o viés médio em cada grupo confirma isso antes de propor qualquer ajuste.
# 
# - Insight principal: a direção/magnitude do viés corresponde à impressão visual para cada modelo?

# %%
actual = ts_holdout[mask]
q25, q75 = actual.quantile([0.25, 0.75])

trough_days = actual[actual <= q25].index
peak_days = actual[actual >= q75].index

for name, forecast in [('Holt-Winters', hw_forecast[mask]), ('SARIMAX', sarimax_forecast[mask])]:
    bias_trough = (forecast[trough_days] - actual[trough_days]).mean()
    bias_peak = (forecast[peak_days] - actual[peak_days]).mean()
    bias_overall = (forecast - actual).mean()
    print(f"{name}:")
    print(f"  Trough days avg bias: {bias_trough:+,.0f}  ({'superprevê' if bias_trough>0 else 'subprevê'})")
    print(f"  Peak days avg bias:   {bias_peak:+,.0f}  ({'superprevê' if bias_peak>0 else 'subprevê'})")
    print(f"  Overall avg bias:     {bias_overall:+,.0f}\n")

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# **Resultados**:
# 
# | | Viés de vale | Viés de pico | Viés geral |
# |---|---|---|---|
# | Holt-Winters | +376,234 | +213,032 | +398,551 |
# | SARIMAX | -199,278 | -370,719 | -38,298 |
# 
# **Holt-Winters**: superprevê em toda a linha, não confinado aos vales como a leitura visual sugeria. 
# O viés geral (+398.551) excede tanto o viés de pico quanto o de vale individualmente, indicando uma superprevisão ampla de nível, com uma questão secundária de amplitude (vales com viés um pouco maior que picos) por cima. 
# **This is primarily a level problem**: a flat downward correction (subtract ~$398K from every previsão, or better, a day-of-week-specific correction using the trough/peak-specific bias) is a reasonable, low-risk adjustment.
# 
# **SARIMAX**: subprevê em toda a linha, mas o viés de pico (-370.719) é mais que o dobro do viés de vale (-199.278) — um problema genuíno de amplitude/forma, não apenas um deslocamento de nível. 
# **Uma correção plana seria o ajuste errado aqui** — adicionar de volta o viés médio supercorrigiria os vales e ainda deixaria os picos substancialmente subprevistos. 
# Isso aponta para uma limitação estrutural, e não algo que um simples ajuste de viés resolve — consistente com a hipótese anterior de que `Seasonal_Index`, treinado em um padrão mais fraco de jan/fev (Passo B), pode estar especificamente suprimindo a capacidade do SARIMAX de alcançar os picos deste período de holdout.
# 

# %% [markdown]
# ## 🔸Passo I: Métricas de Erro no Holdout
# 
# **Objective**: apresentar a avaliação final e honesta de acurácia do modelo selecionado (SARIMAX) no período de holdout, e interpretar o que os erros significam para o uso no negócio.
# 
# **Model**: SARIMAX (1, 0, 1)x(1, 1, 1, 7) X: Feriado/Pagamento/isClosed/Seasonal_Index, selecionado no Passo G pela acurácia do holdout.
# 

# %% [markdown]
# ### Métricas Abrangentes de Erro

# %%
# final_metrics = evaluate_forecast(actual, hw_forecast[mask], 'Holt-Winters (Final, Uncorrected)')
final_metrics = evaluate_forecast(actual, sarimax_forecast[mask], 'SARIMAX (F2, 4-var)')
print(pd.DataFrame([final_metrics]))

# %% [markdown]
# ### Visualização de Previsões vs. Valores Reais

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_train.index[-30:], ts_train.values[-30:], label='Train (last 30 days)', color='gray', alpha=0.6)
ax.plot(actual.index, actual.values, label='Actual (holdout)', color='black', linewidth=1.5)
ax.plot(actual.index, sarimax_forecast[mask].values, label='SARIMAX forecast', color='orange')
ax.legend(); ax.set_title('Holdout: Actual vs. Final Selected Model'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretação de Negócio dos Erros
# 
# - **MAE (416,339)**: on a typical day, the previsão misses by roughly $416K in daily sales, worth contextualizing against average daily sales (~$2.1–2.4M from Step E), this is about **18–21% of a typical day's volume**.
# 
# - **RMSE (497,847)** vs. **MAE**: RMSE é significativamente maior que MAE, consistente com o achado do Passo H de uma distribuição residual de caudas pesadas (curtose=13,3) — um punhado de erros grandes (dias isClosed/feriado) eleva o RMSE mais que o MAE. Essa lacuna em si é informativa: diz que o erro *típico* é menor do que o RMSE sozinho sugeriria, mas os dias de *pior caso* são significativamente piores que o caso típico.
# 
# - **MAPE (16.3%)**: on average, previsãos are off by about 16% of actual sales, a moderate, not excellent, level of accuracy.
# 
# - **Largest errors**: esperados a se concentrar em datas isClosed/feriado (conforme análise residual do Passo H) e possivelmente no período atipicamente forte de jan/fev sinalizado no Passo B — vale uma verificação rápida dos maiores erros individuais por dia para confirmar.
# 
# - **Business acceptability**: 
#     - um erro médio de ~16% é aceitável para decisões de estoque/dimensionamento de equipe neste negócio? 
#     - Provavelmente aceitável para planejamento de alto nível (ex.: faixas semanais de equipe), mas arriscado para compromissos de precisão no nível do dia (ex.: pedidos exatos de estoque).
# 
# - **Known, documented limitations carried into Step J**:
#     - Residuals are not ruído branco (Step H), some structure remains uncaptured.
#     - The model cannot represent `isClosed`/`Pagamento`/`Feriado` explicitly, these dates are the most likely source of the largest errors and must be handled with the planned Step J override (previsão forced to ≈0 on isClosed dates).
# 
# 

# %% [markdown]
# ### Spot-Verificar: Largest Individual Errors
# 
# Ordenando os dias de holdout por erro absoluto e cruzando com `Feriado`, `Pagamento` e `isClosed` — confirmando se os piores erros são explicados por eventos conhecidos (como hipotetizado no Passo H) ou são anomalias inexplicadas.
# 
# - Insight principal: os maiores erros estão concentrados em dias de eventos conhecidos, ou espalhados em datas ordinárias?

# %%
error_detail = pd.DataFrame({
    'Date': actual.index,
    'Actual': actual.values,
    'Forecast': sarimax_forecast[mask].values,
})
error_detail['Error'] = error_detail['Forecast'] - error_detail['Actual']
error_detail['Abs_Error'] = error_detail['Error'].abs()

flags_holdout = df_complete.set_index('Date').loc[actual.index, ['Feriado', 'Pagamento', 'isClosed', 'Vale']].reset_index()
error_detail = error_detail.merge(flags_holdout, on='Date')

top_errors = error_detail.sort_values('Abs_Error', ascending=False).head(10)
print(top_errors.to_string(index=False))

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - **Análise corrigida** (previous version mistakenly used Holt-Winters' previsão, see notebook review): errors are now mixed in direction, not uniformly one-sided. 6 of the top 10 are under-predictions (actual > previsão), concentrated on high-value days (4.5M–5.1M, consistent with Saturdays); 4 are over-predictions, concentrated on lower-value days (~2.0M–2.3M). This is the individual-date evidence behind Step H's peak/trough bias finding: SARIMAX subprevê peaks and superprevê troughs, an amplitude/shape issue, not a uniform level shift.
# 
# - **Sem concentração em dias de evento**: apenas 1 dos top 10 (2021-02-25, `Vale=1`) coincide com uma flag conhecida, e não é notavelmente maior que seus vizinhos sem flag. Os maiores erros não são primariamente impulsionados por Feriado/Pagamento/isClosed/Vale.
# 
# - **Maior erro individual**: 2021-02-06 (-1,33M), uma subprevisão na escala de sábado. Isso substitui a hipótese anterior de "dia-após-isClosed" da análise de Holt-Winters (2021-01-02), que não se aplica aqui; aquele achado específico era específico do modelo e não deve ser carregado para o texto do SARIMAX.
# 
# - **Implicação para o Passo J**: the isClosed override remains valid for its own reason (deterministic zero-sales days), but the bulk of previsão error is tied to SARIMAX's amplitude compression, under-shooting highs, over-shooting lows, rather than to any single explainable calendar event. This is a documented, real limitation of the final model, consistent with Step H's peak-bias hypothesis about `Seasonal_Index` being trained on a weaker Jan/Feb pattern than 2021 actually showed.
# 
# 

# %% [markdown]
# ## 🔸Passo J: Previsão para os Próximos Períodos

# %% [markdown]
# ### Criar Dataset Futuro
# 
# **Objective**: Build the dataset required to generate previsãos for March 2021.
# 
# Como observações futuras ainda não existem, todas as variáveis exógenas usadas pelo modelo SARIMAX final devem ser reconstruídas usando as regras de negócio desenvolvidas ao longo da Parte 1. Isso inclui:
# 
# - Criar uma linha por dia para março de 2021;
# - Identificar feriados (`Feriado`);
# - Sinalizar fechamentos de loja (`isClosed`);
# - Calcular dias de pagamento (`Pagamento`);
# - Mapear o `Seasonal_Index` mensal.
# 
# This dataset will become the input (`exog`) for the previsãoing model.

# %% [markdown]
# #### Datas

# %%
# criar data de início do horizonte futuro
future_start_date = pd.Timestamp(f'2021-03-01')

# criar data de fim do horizonte futuro
future_end_date = pd.Timestamp(f'2021-03-31')

# criar horizonte completo
future_horizon = pd.date_range(start=future_start_date, end=future_end_date, freq='D')

future_horizon

# %%
df_future = pd.DataFrame(future_horizon, columns=['Date'])

print(df_future.describe())

# Criar coluna de departamento
df_future['Departamento'] = choose_dept

# Criar colunas diferentes para datas, mês, ano e combinação
df_future['year'] = df_future['Date'].dt.year
df_future['month'] = df_future['Date'].dt.month
df_future['year_month'] = df_future['Date'].dt.to_period('M')
df_future['weekday'] = df_future['Date'].dt.day_name()


# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# Criado o intervalo de datas para o período de março de 2021.
# - 31 dias nesse período
# - data mínima é 1º de março de 2021
# - data máxima é 31 de março de 2021

# %% [markdown]
# #### `isClosed`

# %%
df_future['isClosed'] = ((df_future['Date'].dt.month == 12) & (df_future['Date'].dt.day == 25)) | \
                          ((df_future['Date'].dt.month == 1) & (df_future['Date'].dt.day == 1))

df_future['isClosed'] = df_future['isClosed'].astype(int) 

print(df_future.sample(10))

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - `isClosed` é sempre em 01-jan e 25-dez
#     - mesmo que nenhuma dessas datas esteja no horizonte futuro que queremos prever, ainda precisamos popular a coluna, pois ela será usada no modelo e todos os valores devem ser 0.
# 

# %% [markdown]
# #### Feriados: `Feriado`
# 
# - Temos que usar `df_holiday_final` criado na Parte 1.
# - Remover *Proc. República Rio Grandense*, pois este feriado não é consistentemente mapeado historicamente.

# %%
# Base de feriados
removed_holiday = ['Proc. República Rio Grandense']

df_holiday_final = df_holiday_filtered[~df_holiday_filtered['nome'].isin(removed_holiday)]

df_holiday_final.sort_values('Date')

# %%
# Mesclar com datas

df_future = df_future.merge(df_holiday_final, how='left', on='Date')

df_future['Feriado'] = df_future['nome'].notna().astype(int)

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - Março de 2021 não apresentou feriados a sinalizar.
# - Analisando historicamente, os feriados de março foram:
#     - Sexta-feira Santa, que não é sempre fixa pois pode cair no final de março, mas é mais frequentemente observada em abril.
#     - Carnaval, também não é um feriado fixo, e é mais frequentemente observado em fevereiro.

# %% [markdown]
# #### `Pagamento`
# 
# - Usar as regras criadas na Parte 1 para obter um possível dia de pagamento.
#     - Função: `get_payment_date(year, month, df_complete)`
# 

# %%
# Calculando o possível dia de pagamento
test_pagamento_date = get_payment_date(2021, 3, df_future)
print(f"Calculated: {test_pagamento_date}") 

# %%
# Adicionando ao dataset futuro
df_pagamento_date = (
    [test_pagamento_date]
    if isinstance(test_pagamento_date, pd.Timestamp)
    else test_pagamento_date
)

df_future['Pagamento'] = df_future['Date'].isin(df_pagamento_date).astype(int)

# %% [markdown]
# ##### ▪️<span style="color:purple">Notas</span>
# 
# - Apenas um dia por mês a ser contabilizado.

# %% [markdown]
# #### `Seasonal_Index`
# 
# - Obter índices dos meses calculados no banco de treino.
#     - tabela: `sazonal_index_train`

# %%
df_future['Seasonal_Index'] = df_future['month'].map(seasonal_index_train)

# %%
print(df_future.head())

# %% [markdown]
# ## Verificar Future Dataset

# %%
print(df_future.describe())

# %% [markdown]
# ## Prevendo Vendas para o Futuro
# 
# - Usar `exog_cols_final` como lista de variáveis exógenas a serem usadas no modelo.
# 

# %% [markdown]
# ### Reajustar o Melhor Modelo nos Dados Completos (Treino + Holdout) e Prever Março de 2021
# 
# Since the holdout period has already served its purpose (Step G/H/I model selection and evaluation), the final model is refit here on the *entire* available history (`df_complete`, through 2021-02-28), this lets previsãoing naturally continue from March 1st.

# %%
if 'Date' in df_complete.columns:
    df_complete_sorted = df_complete.sort_values('Date').set_index('Date').asfreq('D')
else:
    df_complete_sorted = df_complete.sort_index().asfreq('D')

y_full = df_complete_sorted['Sales_adj']
X_full = df_complete_sorted[exog_cols_final] 

model_sarimax_full = sm.tsa.statespace.SARIMAX(
    y_full,
    exog=X_full,
    order=(1, 0, 1),          
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

print(f"Model trained through: {model_sarimax_full.data.dates[-1]}")
print(f"Number of observations: {model_sarimax_full.nobs}")

# %%
print(df_complete.index.name)
print(df_complete.columns.tolist())

# %%
exog_future = df_future.set_index('Date')[exog_cols_final].asfreq('D')

forecast_march = model_sarimax_full.get_forecast(steps=len(exog_future), exog=exog_future)
sales_prediction_march = forecast_march.predicted_mean
conf_int_march = forecast_march.conf_int(alpha=0.05)

print(f"Forecast dates: {sales_prediction_march.index.min()} to {sales_prediction_march.index.max()}")
sales_prediction_march

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - Previous attempt used `model_sarimax` (trained only through 2020-12-31) with March exog, this silently misaligned the dates: `get_previsão(steps=31)` from that model predicts Jan/2021, not March/2021, so the March exog values were being applied to the wrong 31 days.
# 
# - Refitting on the full `df_complete` (through 2021-02-28) fixes this, the next 31 previsão steps now genuinely correspond to March 1–31, 2021.
# 
# 

# %% [markdown]
# ### Visualizar a Previsão de Março de 2021
# 
# Plotting the March previsão alongside the tail end of known history (train + holdout), with the 95% confidence interval shown as a shaded band.

# %%
fig, ax = plt.subplots(figsize=(14, 5))

# Last 60 days of known history for context
history_tail = y_full.iloc[-60:]
ax.plot(history_tail.index, history_tail.values, label='History (Jan-Feb 2021)', color='black', linewidth=1.2)

# March forecast
ax.plot(sales_prediction_march.index, sales_prediction_march.values, label='March 2021 Forecast', color='green', linewidth=1.5)
ax.fill_between(conf_int_march.index, conf_int_march.iloc[:, 0], conf_int_march.iloc[:, 1],
                color='green', alpha=0.15, label='95% CI')

ax.axvline(y_full.index[-1], color='red', linestyle='--', alpha=0.6, label='Forecast start')
ax.legend()
ax.set_title('30-Day Forecast — March 2021')
ax.set_ylabel('Sales (Sales_adj)')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %%
ci_width = conf_int_march.iloc[:, 1] - conf_int_march.iloc[:, 0]
print(ci_width)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ci_width.index, ci_width.values, marker='o')
ax.set_title('95% CI Width by Forecast Day')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# #### ▪️<span style="color:purple">Notas</span>
# 
# - The previsão continue the recent weekly rhythm smoothly from where history left off
# - A banda de confiança se alarga conforme o esperado mais adiante no horizonte
# 

# %% [markdown]
# ### Média Mensal por Ano: Incluindo a Previsão de Março de 2021
# 
# Extending Step E's monthly-average-by-year chart with the March 2021 previsão, so the predicted month can be visually compared against the same month in prior years and against the rest of 2021's actuals.

# %%
combined = pd.concat([y_full, sales_prediction_march])
combined_df = combined.to_frame('Sales_adj')
combined_df['Year'] = combined_df.index.year
combined_df['Month'] = combined_df.index.month

monthly_by_year_full = combined_df.groupby(['Year', 'Month'])['Sales_adj'].mean().unstack(level=0)

fig, ax = plt.subplots(figsize=(12, 5))
for year in monthly_by_year_full.columns:
    style = '--' if year == 2021 else '-'
    marker = 's' if year == 2021 else 'o'
    ax.plot(monthly_by_year_full.index, monthly_by_year_full[year],
            marker=marker, linestyle=style, label=str(year))

ax.axvline(3, color='green', linestyle=':', alpha=0.5)
ax.set_title('Monthly Average Sales by Year (2021 includes March forecast)')
ax.set_xlabel('Month'); ax.set_ylabel('Average Sales_adj')
ax.set_xticks(range(1, 13))
ax.legend(title='Year')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# # 🔹**PARTE 3**: Conclusões
# 
# ---

# %% [markdown]
# ## 🔸Recomendações de Negócio e Próximos Passos
# 

# %%
total_march = sales_prediction_march.sum()
avg_daily_march = sales_prediction_march.mean()
highest_day = sales_prediction_march.idxmax()
highest_value = sales_prediction_march.max()
lowest_day = sales_prediction_march.idxmin()
lowest_value = sales_prediction_march.min()

print(f"Total de vendas previstas para março de 2021: {total_march:,.2f}")
print(f"Média diária prevista: {avg_daily_march:,.2f}")
print(f"Highest sales day: {highest_day.date()} ({highest_day.day_name()}), {highest_value:,.2f}")
print(f"Lowest sales day: {lowest_day.date()} ({lowest_day.day_name()}), {lowest_value:,.2f}")

ci_width_avg = (conf_int_march.iloc[:, 1] - conf_int_march.iloc[:, 0]).mean()
print(f"\nAverage 95% CI width: {ci_width_avg:,.2f}")

# %% [markdown]
# Business insights from the previsão:
# 
# 1. **Volume de vendas previsto**:
#     - Total previsãoed sales for March 2021: **$85,646,050**
#     - Average daily previsão: **$2,762,776**
#     - Dia de maior venda previsto: **2021-03-06 (Saturday)**, $4,357,186. This matches Step A's finding that Saturday is consistently the peak day, a good sign the model correctly extrapolated the weekly pattern into an unseen month, not just memorized treinamento data.
#     - Dia de menor venda previsto: **2021-03-28 (Sunday)**, $2,043,118, consistente com domingo sendo o vale semanal estabelecido.
#     - Peak-to-trough ratio in this previsão: 4,357,186 / 2,043,118 ≈ **2.13x**, próxima, embora ligeiramente abaixo, da razão histórica de 2,24x do Passo A.
# 
# 2. **Avaliação de risco**:
#     - Confidence level: Moderate. The average 95% CI width is **$1,758,262**, roughly 64% of the average daily previsão, a wide band in relative terms. Point previsãos should be treated as a central estimate, not a precise commitment, especially for operational decisions with real cost if wrong in either direction.
#     - Principais pressupostos: `Pagamento`/`Feriado` reconstruídos via regras com 68–76% de acurácia histórica (Parte 1); `Seasonal_Index` derivado apenas dos dados de treino 2018–2020, que o Passo B encontrou mais fracos em jan/fev do que o holdout real de 2021 — vale monitorar se março de 2021 mostra subestimação similar quando os valores reais estiverem disponíveis.
#     - A largura do intervalo de confiança é aproximadamente constante ao longo dos 31 dias (conforme discussão anterior) — consequência da especificação estacionária (`d=0`) do modelo, não uma afirmação de que o final de março é exatamente tão previsível quanto o início de março em todo sentido prático.
# 
# 3. **Recomendações operacionais**:
#     - Estoque: aumentar o estoque antes dos sábados (pico ~R$4,36M) e do dia de pagamento de 4 de março; reduzir para os domingos (vale ~R$2,04M).
#     - Equipe: alinhar maior dimensionamento com sábados e o timing de pagamento do início do mês.
#     - Marketing: considerar promoções leves especificamente aos domingos para suavizar a oscilação de demanda de ~2,13x.
#     - Payment/Voucher planning: coordinate inventory/staffing readiness around the March 4th payday, given `Pagamento`'s confirmed significativo positive effect (Part 1 Mann-Whitney, and F2's SARIMAX coefficient).
#     - Dado o IC amplo, inclua uma margem de segurança nos planos de estoque/equipe em vez de provisionar apenas pela estimativa pontual — trate R$2,76M/dia como o centro de uma faixa significativamente ampla, não como garantia.
# 

# %% [markdown]
# ## 🔸Conclusão
# 
# **Resumo do Projeto**:
# 
# This time series project developed a **SARIMAX(1,0,1)(1,1,1,7)** model, with `Pagamento`, `Feriado`, `Seasonal_Index`, and `isClosed` as exogenous regressors, to previsão daily sales for Depto 4. The model was selected after a structured comparison across the full ARIMA family (ARMA → ARIMA → SARIMA → SARIMAX) and against Holt-Winters exponential smoothing, winning on holdout accuracy once `isClosed` was reinstated as a fitted regressor rather than only as a Step J override.
# 
# Principais achados da análise:
# - Sales follow a strong increasing tendência (+32.8% from 2018 to 2020, corrected from an earlier partial-year estimate) with pronounced weekly sazonalidade, Saturday peaks, Sunday troughs, a ~2.24x historical ratio that the March 2021 previsão reproduced almost exactly (~2.13x).
# - Exogenous business events matter: `Pagamento` (payday) and `Feriado` (holidays) both showed significativo, quantifiable effects once properly controlled for via SARIMAX; `Vale` (voucher days) was tested repeatedly and consistently found not significativo.
# - Os dias `isClosed` exigiram tratamento especial — tanto como regressor ajustado (que melhorou significativamente o ajuste geral do modelo, não apenas a acurácia nos próprios dias fechados) quanto como restrição natural para qualquer data que caia em 25 de dezembro ou 1º de janeiro.
# - The final March 2021 previsão: **$85,646,050 total**, averaging **$2,762,776/day**, with a 95% confidence interval wide enough (~64% of the average previsão) to warrant treating the point estimate as a central planning figure, not a guarantee.
# 
# **Valor de Negócio Entregue**:
# - **Otimização de estoque**: um padrão de demanda orientado por dados, semanal e alinhado ao dia de pagamento, para planejar níveis de estoque, em vez de confiar na intuição.
# - **Eficiência de equipe**: orientação clara sobre quais dias exigem cobertura maior (sábados, dias de pagamento do início do mês) versus menor (domingos).
# - **Proteção de receita**: minimiza o risco de ruptura de estoque nos dias de alta demanda com maior confiança (sábados, dia de pagamento).
# - **Strategic planning**: a documented, testable previsãoing pipeline that can be re-run as new data arrives, rather than a one-off estimate.
# 
# **Limitações Conhecidas** (carregadas das seções anteriores, vale reiterar juntas aqui):
# - As regras de reconstrução de `Pagamento`/`Vale` correspondem a apenas 68–76% dos dias históricos de evento — as flags exógenas futuras carregam incerteza herdada.
# - `Seasonal_Index` was derived from 2018–2020 data; Step B found this treinamento period ran weaker in Jan/Feb than the actual 2021 holdout, a risk that may recur for March if 2021 continues to outperform the historical pattern.
# - Model resíduos were not fully ruído branco even in the final specification, some structure beyond weekly sazonalidade and the four exogenous variables remains unexplained.
# - The previsão's confidence interval doesn't widen meaningfully across the 31-day horizon, a mathematical consequence of the model's stationary (d=0) specification, worth understanding rather than assuming late-March is precisely as certain as early-March in every practical sense.
# 
# **Recomendação Final**: proceed with the SARIMAX previsão for operational planning (inventory, staffing) with the stated confidence band treated as a real planning range, not a formality, and prioritize comparing March 2021's actual results against this previsão once available, both to validate the model and to check whether the Jan/Feb-style sazonal-index mismatch recurs.

# %%



