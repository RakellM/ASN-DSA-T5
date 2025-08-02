import pandas as pd # importando a biblioteca de manipulação de dados
import numpy as np # biblioteca para calculo
import scipy.stats as stats # biblioteca para modelagem
import statsmodels.api as sm # biblioteca para a regressão logística
from matplotlib import pyplot as plt # importando a biblioteca de visualização de dados 
import seaborn as sns# importando a biblioteca de visualização de dados 

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
    while True and var_independente != [] :
        lista_pvalor = []
        lista_variavel = []
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            
            modelo = sm.OLS(Y,X).fit()
            
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
            aic = sm.OLS(Y,X).fit().aic
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
            bic = sm.OLS(Y,X).fit().bic
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
        
        modelo = sm.OLS(Y,X_geral).fit()
        
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
        
        aic_geral = sm.OLS(Y,X_geral).fit().aic
    
        aic_final.append(aic_geral)
        
        preditoras_finais.append(base[var_independente].columns.to_list())
        
        for var in var_independente:
            
            lista_variaveis = var_independente.copy()
            lista_variaveis.remove(var)
            
            X = sm.add_constant(base[ lista_variaveis ])
            aic = sm.OLS(Y,X).fit().aic    
            
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
        
        bic_geral = sm.OLS(Y,X_geral).fit().bic
    
        bic_final.append(bic_geral)
        
        preditoras_finais.append(base[var_independente].columns.to_list())
        
        for var in var_independente:
            
            lista_variaveis = var_independente.copy()
            lista_variaveis.remove(var)
            
            X = sm.add_constant(base[ lista_variaveis ])
            bic = sm.OLS(Y,X).fit().bic    
            
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
        
    # Ajustar a exibição do pandas para não truncar as colunas e linhas longas
    pd.set_option('display.max_colwidth', None)  # Não cortar as colunas
    pd.set_option('display.max_rows', None)  # Mostrar todas as linhas
    
    return resultado


def graficos_var_num(base, variavel):

    """   
    Esta função gera três gráficos (histograma, boxplot e gráfico de violino) 
    para uma variável específica em uma base de dados.
    
    Parâmetros:
        base (pd.DataFrame): Conjunto de dados (DataFrame) contendo as variáveis.
        variavel (str): Nome da variável a ser analisada (string).
    
    Retorna: 
       Três gráficos (histograma, boxplot e violino) exibidos lado a lado para a variável escolhida.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> graficos_var_num(base=df, variavel="BodyFat")
    """
    

    # Definindo o tamanho da figura e criando três subplots em uma linha (1x3)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Histograma
    sns.histplot(data=base, x=variavel, bins=25, ax=axs[0])
    axs[0].set_title(f"Histograma de {variavel}")

    # Boxplot
    sns.boxplot(y=variavel, data=base, ax=axs[1])
    axs[1].set_title(f"Boxplot de {variavel}")

    # Gráfico de violino
    sns.violinplot(y=variavel, data=base, ax=axs[2])
    axs[2].set_title(f"Gráfico de Violino de {variavel}")

    # Ajustar o layout para não sobrepor os títulos
    plt.tight_layout()
    plt.show()