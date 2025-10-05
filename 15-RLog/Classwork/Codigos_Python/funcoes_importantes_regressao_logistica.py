import pandas as pd # importando a biblioteca de manipulação de dados
import numpy as np # biblioteca para calculo
import scipy.stats as stats # biblioteca para modelagem
import statsmodels.api as sm # biblioteca para a regressão logística
import statistics
import seaborn as sns
import math
import matplotlib.pyplot as plt

def selecionar_pvalor_forward(var_dependente, var_independente, base, signif):
    """   
    Esta função realiza uma seleção forward stepwise com base no p-valor das variáveis independentes.
    A cada passo, adiciona a variável independente com o menor p-valor ao modelo, desde que o p-valor 
    seja menor que o nível de significância especificado.
    
    Parâmetros:
    var_dependente (str): Nome da variável dependente.
    var_independente (list): Lista de variáveis independentes a serem avaliadas.
    base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    signif (float): Nível de significância para a inclusão das variáveis (por exemplo, 0.05).
    
    Retorna: 
    pd.DataFrame: DataFrame contendo as variáveis selecionadas e seus respectivos p-valores.
    
    Exemplo de uso:
        >>> import pandas as pd
        >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
        >>> colunas_pvalor = selecionar_pvalor_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df, signif=0.05)
        >>> colunas_pvalor
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    

    preditoras = []
    pvalor_preditoras = []
    Y = base[var_dependente]
    while True and var_independente != []:
        lista_pvalor = []
        lista_variavel = []
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            
            modelo = sm.GLM(Y,X,family=sm.families.Binomial()).fit()
            
            if( preditoras == []):    
                
                pvalor = modelo.pvalues[1]
                variavel = modelo.pvalues.index[1]
            
            else:
                
                pvalor = modelo.pvalues.drop(preditoras)[1]
                variavel = modelo.pvalues.drop(preditoras).index[1]
                
            lista_pvalor.append(pvalor)
            lista_variavel.append(variavel)          
        
        if( lista_pvalor[ np.argmin(lista_pvalor) ] < signif ):
            preditoras.append( lista_variavel[np.argmin(lista_pvalor)] )
            pvalor_preditoras.append(lista_pvalor[ np.argmin(lista_pvalor) ])
            var_independente.remove( lista_variavel[ np.argmin(lista_pvalor)] )
        else:
            break
    info_final = pd.DataFrame({ 'var': preditoras, 'pvalor': pvalor_preditoras})
    return info_final


def selecionar_aic_forward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção forward stepwise com base no critério de informação de Akaike (AIC).
    A cada passo, adiciona a variável independente que minimiza o AIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos AICs, 
        ordenados do menor para o maior AIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_aicforward = selecionar_aic_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_aicforward
    
    criada por Mateus Rocha - time ASN.Rocks
    """    

    preditoras = []
    aic_preditoras = []
    Y = base[var_dependente]
    lista_final = []
    aic_melhor = float('inf')
    
    while True and var_independente != []:
        lista_aic = []
        lista_variavel = []
        lista_modelos =[]
        if(var_independente == []):
            break
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            aic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().aic
            variavel = var
                
            lista_aic.append(aic)
            
            lista_variavel.append(var)
            
            lista_modelos.append( [var] +  preditoras )
            
        if( lista_aic[ np.argmin(lista_aic) ] < aic_melhor ):
            
            lista_final.append(lista_modelos[ np.argmin(lista_aic)]  )
            
            preditoras.append( lista_variavel[np.argmin(lista_aic)] )
            
            aic_preditoras.append(lista_aic[ np.argmin(lista_aic) ])
            
            var_independente.remove( lista_variavel[ np.argmin(lista_aic)] )
            
            aic_melhor = lista_aic[ np.argmin(lista_aic) ] 
            
        else:
            break
        
    info_final = pd.DataFrame({ 'var': lista_final, 'aic': aic_preditoras}).sort_values(by = 'aic')
    return info_final


def selecionar_bic_forward(var_dependente, var_independente, base):
    
    """   
    Esta função realiza uma seleção forward stepwise com base no critério de informação bayesiano (BIC).
    A cada passo, adiciona a variável independente que minimiza o BIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos BICs, 
        ordenados do menor para o maior BIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_bicforward = selecionar_bic_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_bicforward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    preditoras = []
    bic_preditoras = []
    Y = base[var_dependente]
    lista_final = []
    bic_melhor = float('inf')
    
    while True and var_independente != []:
        lista_bic = []
        lista_variavel = []
        lista_modelos =[]
        if(var_independente == []):
            break
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            bic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().bic
            variavel = var
                
            lista_bic.append(bic)
            
            lista_variavel.append(var)
            
            lista_modelos.append( [var] +  preditoras )
            
        if( lista_bic[ np.argmin(lista_bic) ] < bic_melhor ):
            
            lista_final.append(lista_modelos[ np.argmin(lista_bic)]  )
            
            preditoras.append( lista_variavel[np.argmin(lista_bic)] )
            
            bic_preditoras.append(lista_bic[ np.argmin(lista_bic) ])
            
            var_independente.remove( lista_variavel[ np.argmin(lista_bic)] )
            
            aic_melhor = lista_bic[ np.argmin(lista_bic) ] 
            
        else:
            break
        
    info_final = pd.DataFrame({ 'var': lista_final, 'bic': bic_preditoras}).sort_values(by = 'bic')
    return info_final


def selecionar_pvalor_backward(var_dependente, var_independente, base, signif):
    """   
    Esta função realiza uma seleção backward stepwise com base no p-valor das variáveis independentes.
    A cada passo, remove a variável independente com o maior p-valor do modelo, 
    desde que seja maior que o nível de significância especificado.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      signif (float): Nível de significância para a inclusão das variáveis (por exemplo, 0.05).
      
    Retorna: 
        pd.DataFrame: DataFrame contendo as variáveis restantes após a seleção backward.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_pvalorbackward = selecionar_pvalor_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), signif = 0.05 ,base=df)
            >>> colunas_pvalorbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    Y = base[var_dependente]
    
    while True and var_independente != []:
        
        X_geral = sm.add_constant(base[var_independente])
        
        modelo = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit()
        
        pvalor_geral = modelo.pvalues
        
        variavel_geral = modelo.pvalues.index
        
        if(pvalor_geral[ np.argmax(pvalor_geral) ] > signif ):
            var_independente.remove( variavel_geral[ np.argmax(pvalor_geral) ] )
        else:
            break
    
    
    
    info_final = pd.DataFrame({ 'var': var_independente})
    return info_final


def selecionar_aic_backward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção backward stepwise com base no critério de informação de Akaike (AIC).
    A cada passo, adiciona a variável independente que minimiza o AIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos AICs, 
        ordenados do menor para o maior AIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_aicbackward = selecionar_aic_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_aicbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    Y = base[var_dependente]
    
    preditoras_finais = []
    
    aic_final = []
    
    while True and var_independente != []:
        
        lista_aic = []
        lista_preditoras = []

        X_geral = sm.add_constant(base[var_independente])
        
        aic_geral = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit().aic
    
        aic_final.append(aic_geral)
        
        preditoras_finais.append(base[var_independente].columns.to_list())
        
        for var in var_independente:
            
            lista_variaveis = var_independente.copy()
            lista_variaveis.remove(var)
            
            X = sm.add_constant(base[ lista_variaveis ])
            aic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().aic    
            
            lista_aic.append(aic)
            
            lista_preditoras.append(var)
            
        if(lista_aic[ np.argmin(lista_aic) ] < aic_geral ):
            var_independente.remove( lista_preditoras[ np.argmin(lista_aic) ] )
            
        else:
            break
    
    
    info_final = pd.DataFrame({ 'var': preditoras_finais, 'aic':aic_final }).sort_values(by = 'aic')
    return info_final

def selecionar_bic_backward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção backward stepwise com base no critério de informação bayesiano (BIC).
    A cada passo, adiciona a variável independente que minimiza o BIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos BICs, 
        ordenados do menor para o maior BIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_bicbackward = selecionar_bic_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_bicbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    Y = base[var_dependente]
    
    preditoras_finais = []
    
    bic_final = []
    
    while True and var_independente != []:
        
        lista_bic = []
        lista_preditoras = []

        X_geral = sm.add_constant(base[var_independente])
        
        bic_geral = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit().bic
    
        bic_final.append(bic_geral)
        
        preditoras_finais.append(base[var_independente].columns.to_list())
        
        for var in var_independente:
            
            lista_variaveis = var_independente.copy()
            lista_variaveis.remove(var)
            
            X = sm.add_constant(base[ lista_variaveis ])
            bic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().bic    
            
            lista_bic.append(bic)
            
            lista_preditoras.append(var)
            
        if(lista_bic[ np.argmin(lista_bic) ] < bic_geral ):
            var_independente.remove( lista_preditoras[ np.argmin(lista_bic) ] )
            
        else:
            break
    
    
    info_final = pd.DataFrame({ 'var': preditoras_finais, 'bic':bic_final }).sort_values(by = 'bic')
    return info_final

def stepwise( var_dependente , var_independente , base, metrica, signif = 0.05, epsilon = 0.0001):
      
    """   
    Esta função realiza a seleção stepwise de variáveis, usando os métodos forward e backward 
    com base em uma métrica específica (AIC, BIC ou p-valor).
    O processo consiste em primeiro aplicar a seleção forward com a métrica escolhida e, 
    em seguida, a backward, ajustando o modelo até que a diferença entre as métricas seja menor 
    que um valor de tolerância (epsilon).
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      metrica (str): A métrica a ser usada no processo de seleção (pode ser 'aic', 'bic', ou 'pvalor').
      signif (float): Nível de significância usado para a seleção por p-valor (padrão 0.05).
      epsilon (float): Diferença mínima aceitável entre as métricas forward e backward para parar o processo (padrão 0.0001).
    Retorna: 
         Resultado da seleção de variáveis com base no método e métrica escolhidos.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_stepwise = stepwise(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base = df ,metrica='aic', signif=0.05)
            >>> colunas_stepwise
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    lista_var = var_independente
    
    metrica_forward = 0
    
    metrica_backward = 0
    
    while True:
    
        if(metrica == 'aic'):
            resultado = selecionar_aic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)

            if (len(resultado) == 1):
                return resultado
            
            resultado_final = selecionar_aic_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list()[0], base = base)

            if(len(resultado_final) == 1):
                return resultado_final

            metrica_forward = resultado['aic'].to_list()[0]

            metrica_backward = resultado_final['aic'].to_list()[0]


        elif(metrica == 'bic'):
            resultado = selecionar_bic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)

            if (len(resultado) == 1):
                return resultado

            resultado_final = selecionar_bic_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list()[0], base = base)

            if(len(resultado_final) == 1):
                return resultado_final

            metrica_forward = resultado['bic'].to_list()[0]

            metrica_backward = resultado_final['bic'].to_list()[0]

        elif(metrica == 'pvalor'):
            resultado = selecionar_pvalor_forward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)

            if (len(resultado) == 1):
                return resultado

            resultado_final = selecionar_pvalor_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list(), base = base, signif = signif)

            if(len(resultado_final) == 1):
                return resultado_final

            return resultado_final

        if( abs(metrica_forward - metrica_backward) < epsilon ):
            break
        else:
            var_independente = set(resultado_final['var'].to_list() + lista_var)

    
def step( var_dependente , var_independente , base, metodo, metrica, signif = 0.05):
    """   
    Esta função realiza a seleção de variáveis usando os métodos forward, backward ou stepwise, 
    com base em uma métrica escolhida (AIC, BIC ou p-valor).O usuário pode escolher o método de 
    seleção (forward, backward ou both) e a métrica desejada para o critério de inclusão ou exclusão de variáveis.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      metrica (str): A métrica a ser usada no processo de seleção (pode ser 'aic', 'bic', ou 'pvalor').
      metodo (str): Método de seleção ('forward', 'backward' ou 'both').
      signif (float): Nível de significância usado para a seleção por p-valor (padrão 0.05).
    Retorna: 
        Resultado da seleção de variáveis com base no método e métrica escolhidos.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_step = step(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base = df, metodo = 'forward' ,metrica='aic', signif=0.05)
            >>> colunas_step
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    if( metodo == 'forward' and metrica == 'aic' ):
        resultado = selecionar_aic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'forward' and metrica == 'bic' ):
        resultado = selecionar_bic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'forward' and metrica == 'pvalor' ):
        resultado = selecionar_pvalor_forward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)
    elif( metodo == 'backward' and metrica == 'aic' ):
        resultado = selecionar_aic_backward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'backward'and metrica == 'bic' ):
        resultado = selecionar_bic_backward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'backward' and metrica == 'pvalor' ):
        resultado = selecionar_pvalor_backward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)
    elif(metodo == 'both'):
        resultado = stepwise( var_dependente = var_dependente , var_independente = var_independente , base = base, metrica = metrica, signif = signif)
        
    return resultado




def univariada_variavel_numerica(dado, variavel):
    """
    Gera uma matriz de gráficos (2x2) para uma variável contínua.

    [1,1] Histograma
    [1,2] Gráfico de violino
    [2,1] Box plot
    [2,2] Box plot com pontos sobrepostos

    Acima dos gráficos, exibe uma tabela com as estatísticas descritivas da variável.

    Parâmetros:
        dado (pd.DataFrame): Base de dados contendo a variável
        variavel (str): Nome da variável a ser analisada

    Retorna:
        None

    Exemplo de uso:
        >> dado = pd.DataFrame({"variavel_exemplo": np.random.normal(loc=50, scale=10, size=100)})
        >>univariada_variavel_numerica(dado, "variavel_exemplo")
        
    """
    
    # Calcular as estatísticas descritivas
    desc_stats = dado[variavel].describe().to_frame().T
    desc_stats = desc_stats.round(4)  # Limitar a 4 casas decimais

    # Configuração dos subplots
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Análise da variável: {variavel}", fontsize=16, y=0.98)

    # Adicionar a tabela no topo
    ax_table = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_table.axis("off")
    table = ax_table.table(cellText=desc_stats.values,
                           colLabels=desc_stats.columns,
                           rowLabels=desc_stats.index,
                           cellLoc="center",
                           loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(desc_stats.columns))))

    # [1,1] Histograma
    ax1 = plt.subplot2grid((3, 2), (1, 0))
    sns.histplot(dado[variavel], kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Histograma", fontsize=12)
    ax1.set_xlabel(variavel)

    # [1,2] Gráfico de violino
    ax2 = plt.subplot2grid((3, 2), (1, 1), sharex=ax1)
    sns.violinplot(x=dado[variavel], ax=ax2, color="lightgreen")
    ax2.set_title("Gráfico de violino", fontsize=12)
    ax2.set_xlabel(variavel)

    # [2,1] Box plot
    ax3 = plt.subplot2grid((3, 2), (2, 0), sharex=ax1)
    sns.boxplot(x=dado[variavel], ax=ax3, color="orange")
    ax3.set_title("Box plot", fontsize=12)
    ax3.set_xlabel(variavel)

    # [2,2] Box plot com pontos sobrepostos
    ax4 = plt.subplot2grid((3, 2), (2, 1), sharex=ax1)
    sns.boxplot(x=dado[variavel], ax=ax4, color="lightcoral")
    sns.stripplot(x=dado[variavel], ax=ax4, color="black", alpha=0.5, jitter=True)
    ax4.set_title("Box plot com pontos", fontsize=12)
    ax4.set_xlabel(variavel)

    # Ajustes finais
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def univariada_variavel_categoria(dado, variavel):
    """
    Análises para variáveis categóricas.

    1. Retorna o describe transposto e formatado em uma tabela.
    2. Retorna uma tabela com a frequência de cada nível (incluindo percentuais e total).
    3. Plota um gráfico de barras com a frequência e exibe os valores no topo.

    Parâmetros:
        dado (pd.DataFrame): O dataframe contendo os dados.
        variavel (str): O nome da variável categórica para análise.

    Retorna:
        None

    Exemplo de uso:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'Categoria': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'A', 'B']})
        >>> univariada_variavel_categoria(df, 'Categoria')
    """
    # Verificar se a variável está no DataFrame
    if variavel not in dado.columns:
        raise ValueError(f"A variável '{variavel}' não está no DataFrame.")

    # 1. Describe transposto e formatado
    describe_table = dado[variavel].describe().to_frame()
    describe_table = describe_table.T
    describe_table.index = [variavel]

    # Exibir a tabela formatada
    print("Describe da variável categórica:")
    display(describe_table)

    # 2. Frequência de cada nível (com percentuais e total)
    frequency_table = dado[variavel].value_counts().reset_index()
    frequency_table.columns = [variavel, 'Frequência']
    frequency_table['Percentual (%)'] = (frequency_table['Frequência'] / len(dado) * 100).round(2)

    # Adicionar uma linha para o total
    total_row = pd.DataFrame({
        variavel: ['Total'],
        'Frequência': [frequency_table['Frequência'].sum()],
        'Percentual (%)': [100.0]
    })
    frequency_table = pd.concat([frequency_table, total_row], ignore_index=True)

    # Exibir a tabela formatada
    print("Tabela de frequência da variável categórica (com percentuais e total):")
    display(frequency_table)

    # 3. Gráfico de barras com frequência
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=variavel, y='Frequência', data=frequency_table[:-1], errorbar=None)

    # Adicionar os valores no topo das barras
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')

    # Configurar o gráfico
    plt.title(f'Gráfico de Frequência: {variavel}')
    plt.xlabel(variavel)
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analise_var_numerica_por_percentil(data, x, y, q=10, grafico='none'):
    """
    Ordena a variável x, divide em percentis e sumariza estatísticas.

    Parâmetros:
        data (pd.DataFrame): O banco de dados contendo as variáveis.
        x (str): O nome da variável independente (explanatória).
        y (str): O nome da variável dependente (resposta).
        q (int): O número de percentis (default: 10).
        grafico (str): Opção de gráfico: 'p', 'logito', 'ambos', 'none' (default: 'none').

    Retorno:
        pd.DataFrame: DataFrame com as estatísticas por percentil, incluindo:
                      - Percentil
                      - n (número de linhas)
                      - Min de x
                      - Max de x
                      - p (média de y)
                      - logito de p

    Exemplo de uso
        >> data = pd.DataFrame({'x': np.random.uniform(0, 100, 1000), 
        'y': np.random.randint(0, 2, 1000)})
        >> resultado = analise_var_numerica_por_percentil(data, 'x', 'y', q=10, grafico='ambos')
        >> print(resultado)
    """
    # Certificar-se de que a variável y está no formato numérico
    data[y] = pd.to_numeric(data[y], errors='coerce')

    # Ordenar os dados pela variável x
    data = data.sort_values(by=x).reset_index(drop=True)

    # Criar os percentis
    data['percentil'] = pd.qcut(data[x], q=q, labels=[str(i) for i in range(1, q + 1)])

    # Sumarizar as estatísticas por percentil
    summary = data.groupby('percentil').agg(
        n=(x, 'count'),
        min_x=(x, 'min'),
        max_x=(x, 'max'),
        p=(y, 'mean')
    ).reset_index()

    # Calcular o logito de p
    summary['logito_p'] = np.log(summary['p'] / (1 - summary['p']))

    # Ajuste para lidar com casos onde p é 0 ou 1
    epsilon = 1e-10  # Pequeno valor para ajustar 0 e 1
    summary['logito_p'] = np.log(np.clip(summary['p'], epsilon, 1 - epsilon) / 
                                 (1 - np.clip(summary['p'], epsilon, 1 - epsilon)))


    # Opções de gráfico
    if grafico in ['p', 'logito', 'ambos']:
        plt.figure(figsize=(12, 6))

        if grafico == 'p':
            plt.scatter(summary['percentil'], summary['p'], color='blue')
            plt.title('Gráfico de Percentil x p')
            plt.xlabel('Percentil')
            plt.ylabel('p (média de y)')
            plt.grid(True)
            plt.show()

        elif grafico == 'logito':
            plt.scatter(summary['percentil'], summary['logito_p'], color='red')
            plt.title('Gráfico de Percentil x Logito de p')
            plt.xlabel('Percentil')
            plt.ylabel('Logito de p')
            plt.grid(True)
            plt.show()

        elif grafico == 'ambos':
            # Gráficos lado a lado
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

            # Gráfico Percentil x p
            axes[0].scatter(summary['percentil'], summary['p'], color='blue')
            axes[0].set_title('Percentil x p')
            axes[0].set_xlabel('Percentil')
            axes[0].set_ylabel('p (média de y)')
            axes[0].grid(True)

            # Gráfico Percentil x Logito de p
            axes[1].scatter(summary['percentil'], summary['logito_p'], color='red')
            axes[1].set_title('Percentil x Logito de p')
            axes[1].set_xlabel('Percentil')
            axes[1].set_ylabel('Logito de p')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

    return summary