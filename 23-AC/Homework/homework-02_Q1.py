# %%
# LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import tanglegram

# %%
# FUNCTION 1
## Same Scale
#Esta função coloca os dados na mesma escala usando a equacao padrao
# (x-xbarra)/sd
# só criei isso porque o Scaler do python divide pelo desvio padrao populacional e não amostral
# em conjuntos não tão grande, isso dá muita diferença
#Parâmetros:
# -data: Conjunto de dados.
#Retorna: Conjunto de dados na mesma escala.

def mesma_escala(data):
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=1)
    return data

# Exemplo de uso
# mesma_escala(mcdonalds)

# %% 
# FUNCTION 2
## Create Dendrogram
#Esta função cria o gráfico do dendograma conforme o método escolhido
#Parâmetros:
# -data: Conjunto de dados.
# -metodo: Método de ligação a ser utilizado na clusterização hierárquica (ex: 'average', 'ward', 'complete', 'single', 'centroid').
# -padronizar: Se seus dados não precisam ser padronizados escolher False, caso contrário, True
#Retorna:Apenas exibe o dendrograma.

def fazer_dendograma(data, metodo, padronizar):
    # Padronizando os dados, se padronizar for True
    if padronizar:
        def mesma_escala(data):
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=1)
            return data
        base_padronizada = mesma_escala(data)
#        print("Base padronizada:")
#        print(base_padronizada.head())
        
        # Calculando a matriz de distâncias com os dados padronizados
        d = pdist(base_padronizada, metric='euclidean')
#        print("Matriz de distâncias (padronizada):")
        matriz = squareform(d)
    else:
#        print("Utilizando a base original:")
#        print(data.head())
        
        # Calculando a matriz de distâncias com os dados originais
        d = pdist(data, metric='euclidean')
#        print("Matriz de distâncias (original):")
        matriz = squareform(d)
    
    # Aplicando o linkage (clustering hierárquico)
    cluster_hierarquico = linkage(d, method=metodo)
    
    # Plotando o dendrograma
    plt.figure(figsize=(5, 3))
    plt.title(f"Dendrograma usando {metodo.capitalize()} linkage")
    
    # Exibindo o dendrograma com os rótulos da base de dados
    dendrogram(cluster_hierarquico, labels=data.index)
    
    # Rotacionando os rótulos no eixo x
    plt.xticks(rotation=90)
    
    # Exibindo o gráfico
    plt.show()

# Exemplo de uso
# fazer_dendograma(mcdonalds, metodo='single', padronizar=True)

# %%
# FUNCTION 3
## SQT calculation
#Esta função calcula a soma dos quadrados totais (SQT) para cada variável dentro de cada cluster.
# Retorna o calculo passo a passo e uma tabela com a SQT de cada variável,
# além do valor total da soma dos quadrados.
#Parâmetros:
#    - data: Conjunto de dados.
#    - grupos: Um vetor que indica a qual cluster cada observação pertence.
#    - verbose: Se True, imprime informações detalhadas sobre o cálculo.
#Retorna:
#    - Uma tabela com a SQT de cada variável por cluster.
#    - O valor total da soma dos quadrados (WSS).

def calcular_soma_quadrados_totais(data, grupos, verbose):

    sqt_total = 0
    resultado = []

    if verbose:
        print(f"Dados recebidos: \n{data}")
        print(f"Grupos recebidos: \n{grupos}")
    
    # Percorrendo cluster a cluster (grupo a grupo)
    for cluster_gerado in np.unique(grupos):
        if verbose:
            print(f"\nProcessando o cluster {cluster_gerado}...")
        
        # Filtrando os dados do cluster atual
        cluster_data = data[grupos == cluster_gerado]
        
        if verbose:
            print(f"Dados do cluster {cluster_gerado}: \n{cluster_data}")
        
        # Verificando se o cluster tem pelo menos 1 ponto
        if len(cluster_data) > 0:
            # Calculando o centroide (média de cada variável)
            centroide = np.mean(cluster_data, axis=0)
            if verbose:
                print(f"Centroide do cluster {cluster_gerado}: \n{centroide}")
            
            # Calculando a soma dos quadrados para cada variável
            sqt_variaveis = np.sum((cluster_data - centroide) ** 2, axis=0)
            if verbose:
                print(f"SQT das variáveis para o cluster {cluster_gerado}: \n{sqt_variaveis}")
            
            # Armazenando o resultado para cada cluster e variável
            resultado.append(sqt_variaveis)
            
            # Soma dos quadrados dentro do cluster (WSS para o cluster)
            sqt_total += np.sum(np.linalg.norm(cluster_data - centroide, axis=1)**2)
        else:
            resultado.append([np.nan] * data.shape[1])  # Caso de cluster vazio
    
    # Convertendo o resultado para um DataFrame
    df_resultado = pd.DataFrame(np.vstack(resultado), columns=[f'Variável {i+1}' for i in range(data.shape[1])])
    df_resultado.index = [f'Cluster {int(c)}' for c in np.unique(grupos)]
    
    if verbose:
        # Imprimindo o valor total da soma dos quadrados
        print(f"\nSoma total dos quadrados (WSS): {sqt_total}")
    
    return df_resultado, sqt_total

# Exemplo de uso
#roda o método que te interessa
#mcdonalds_padronizado = mesma_escala(mcdonalds)
#d = pdist(mcdonalds_padronizado, metric='euclidean')
#cluster_hierarquico = linkage(d, method='single')
#marca os grupos
#grupos = fcluster(cluster_hierarquico , t=2, criterion='maxclust')
#grupos
#chama a função
#df_resultado, wss_total = calcular_soma_quadrados_totais(mcdonalds_padronizado, grupos, verbose=True)
# Exibir a tabela de resultados
#print(df_resultado)
#print(f"\nSoma total dos quadrados (WSS):", wss_total)
#print(grupos)`

# %%
# FUNCTION 4
## Elbow Method
#Esta função faz o cluster hierárquico com o método definido, calcula a soma dos quadrados totais (SQT)
#para diferentes números de clusters e plota o gráfico do método do cotovelo (Elbow).
#Parâmetros:
# - data: Conjunto de dados.
# - metodo: Método de ligação a ser utilizado na clusterização hierárquica (ex: 'average', 'ward', 'complete', 'single', 'centroid').
# - max_clusters: O número máximo de clusters a ser considerado no cálculo, ou seja, vai calcular para todos os clusters até este valor.
# - padronizar: Se seus dados não precisam ser padronizados escolher False, caso contrário, True
# importante documentar que se os dados forem padronizados (necessário quando tem escalas diferentes)
# o SQT será calculado com os dados padronizados.
#Retorna:
#    - Plota o gráfico do método do cotovelo para o número de clusters.

def grafico_elbow(data, metodo, max_clusters, padronizar):
    # Verifica se os dados devem ser padronizados
    if padronizar:
        def mesma_escala(data):
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=1)
            return data
        data_utilizado = mesma_escala(data)
    else:
        print("Usando os dados originais...")
        data_utilizado = data
    
    # Criar o linkage (cluster_hierarquico) com o método especificado
    cluster_hierarquico = linkage(data_utilizado, method=metodo)
    
    # Lista para armazenar os valores de SQT
    sqt = []
    
    # Calcular SQT para diferentes números de clusters
    for n_clusters in range(1, max_clusters + 1):
        grupos = fcluster(cluster_hierarquico, t=n_clusters, criterion='maxclust')
        
        if n_clusters > 1:
            # A função 'calcular_soma_quadrados_totais' retorna dois valores, o WSS 
            # está na segunda posição
            _, wss_total = calcular_soma_quadrados_totais(data_utilizado, grupos, verbose=False)
            sqt.append(wss_total)
        else:
            sqt.append(np.nan)  # Para o caso de n_clusters = 1
    
    # Plotar o gráfico do método do cotovelo (Elbow)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), sqt, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma dos Quadrados das Distâncias')
    plt.title(f'Elbow para Clustering Hierárquico pelo método {metodo.capitalize()}')
    plt.show()

# Exemplo de uso
#grafico_elbow(mcdonalds, metodo='average', max_clusters=10, padronizar = False)

# %%
# FUNCTION 5
## Sillhouette Method
#Esta função faz o cluster hierárquico com o método definido e calcula o a_i, b_i, s_i, s_k para cada k cluster criados
#a ideia aqui é ter uma função que faz todas as contas necessárias para silhueta, a fim de armazenar um df.
#Parâmetros:
# - data: Conjunto de dados.
# - metodo: Método de ligação a ser utilizado na clusterização hierárquica (ex: 'average', 'ward', 'complete', 'single', 'centroid').
# - max_clusters: O número máximo de clusters a ser considerado no cálculo, ou seja, vai calcular para todos os clusters até este valor.
# - padronizar: Se seus dados não precisam ser padronizados escolher False, caso contrário, True
# importante documentar que se os dados forem padronizados (necessário quando tem escalas diferentes)
# a silhueta será calculado com os dados padronizados.
#Retorna:
#    - df_silhueta: DataFrame contendo os resultados (label_ponto, grupo, a_i, b_i, s_i, s_k, k).

def s_silhouette(data, metodo, max_clusters, padronizar):

    # Se padronizar for True, padroniza os dados
    if padronizar:
        def mesma_escala(data):
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=1)
            return data
        data_padronizada = mesma_escala(data)
        d = pdist(data_padronizada, metric='euclidean')
        print("Dados padronizados para clustering")
    else:
        d = pdist(data, metric='euclidean')
        print("Dados originais utilizados para clustering")
    
    # Convertendo a matriz de distâncias para formato quadrado (necessária para silhouette)
    d_square = squareform(d)
    
    # Criando a lista para armazenar os resultados de cada k
    silhueta = []
    
    # Rodando o clustering hierárquico uma vez
    cluster_hierarquico = linkage(d, method=metodo)

    # Iterando para cada valor de k (de 2 até max_clusters)
    for k in range(2, max_clusters + 1):
        grupos = fcluster(cluster_hierarquico, t=k, criterion='maxclust')

        # Calculando a e b para cada ponto usando a função calc_a_b
        a_vals, b_vals = calc_a_b(d_square, grupos, np.unique(grupos))

        # Calculando silhouette para cada ponto, com a regra de cluster único
        silhouette_vals = []
        for a_i, b_i, grupo in zip(a_vals, b_vals, grupos):
            # Se o ponto está sozinho no cluster, defina silhueta como 0
            if np.sum(grupos == grupo) == 1:
                silhouette_vals.append(0)
            else:
                silhouette_vals.append((b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0)

        # Calculando a média das silhuetas
        silhouette_avg = np.mean(silhouette_vals)

        # Criando DataFrame temporário para armazenar os resultados
        df_k = pd.DataFrame({
            'label_ponto': data.index,
            'grupo': grupos,
            'a_i': a_vals,
            'b_i': b_vals,
            's_i': silhouette_vals,
            's_k': silhouette_avg,
            'k': [k] * len(grupos)  # Adiciona o valor de k em todas as linhas
        })

        # Adiciona o DataFrame resultante na lista de resultados
        silhueta.append(df_k)
    
    # Concatenando todos os DataFrames gerados para cada valor de k
    df_silhueta = pd.concat(silhueta, ignore_index=True)

    return df_silhueta


# Função para calcular as distâncias intra-cluster (a) e inter-cluster (b)
def calc_a_b(d_matrix, clusters, cluster_id):
    a_values = []
    b_values = []
    for i in range(len(clusters)):
        same_cluster = np.where(clusters == clusters[i])[0]
        other_clusters = np.where(clusters != clusters[i])[0]
        
        # Distância média para o mesmo cluster (a)
        if len(same_cluster) > 1:
            a = np.mean([d_matrix[i, j] for j in same_cluster if i != j])
        else:
            a = 0  # Se não houver outro ponto no cluster, a = 0
        a_values.append(a)
        
        # Distância média para o cluster mais próximo (b)
        b = np.min([np.mean([d_matrix[i, j] for j in np.where(clusters == c)[0]]) 
                    for c in np.unique(clusters) if c != clusters[i]])
        b_values.append(b)
    
    return a_values, b_values

# Exemplo de uso
#df_silhueta = s_silhouette(mcdonalds, metodo='average', max_clusters=11, padronizar=True)
#df_silhueta

# %%
# FUNCTION 6
## Silhouette Plot
#Esta função cria o gráfico da silhueta para o s de cada ponto
#Parâmetros:
#  df_silhueta : DataFrame obtido na função anterior contendo os resultados de silhueta para cada número de clusters.
#Retorna: Apenas exibe o gráfico da silhueta.

def grafico_silhouette_por_ponto(df_silhueta):
    # Obter os valores únicos de k
    k_values = df_silhueta['k'].unique()
    
    # Definir o número de colunas para o layout dos gráficos
    num_cols = len(k_values)
    
    # Criar um gráfico para cada valor de k
    fig, axes = plt.subplots(1, num_cols, figsize=(15, 6), sharey=False)
    
    # Caso tenha apenas 1 valor de k, evitar erro na iteração
    if num_cols == 1:
        axes = [axes]  # Transformar o único eixo em lista

    for i, k in enumerate(k_values):
        # Filtrar o dataframe para o valor de k
        df_k = df_silhueta[df_silhueta['k'] == k]
        
        # Criar gráfico de barras horizontais sem ordenar os valores
        axes[i].barh(df_k['label_ponto'], df_k['s_i'], color='lightblue', edgecolor='black')
        
        # Adicionar linha pontilhada para o valor de s_k
        axes[i].axvline(x=df_k['s_k'].values[0], color='red', linestyle='--', label=f's_k = {df_k["s_k"].values[0]:.3f}')
        
        # Adicionar linha pontilhada para a média
        axes[i].axvline(x=0, color='blue', linestyle='-')
        
        # Definir título e labels
        axes[i].set_title(f'k = {k}')
        axes[i].set_xlabel('Valor de s_i')
        
        # Mostrar as labels do eixo y para todos os gráficos
#        axes[i].set_yticks(df_k['label_ponto'])
#        axes[i].set_yticklabels(df_k['label_ponto'])
 
        # Mostrar as labels do eixo Y apenas no primeiro gráfico
        if i == 0:
            axes[i].set_yticks(df_k['label_ponto'])
            axes[i].set_yticklabels(df_k['label_ponto'])
        else:
            axes[i].set_yticks([])
            axes[i].set_yticklabels([])
            
    # Ajustar layout
    plt.tight_layout()
    plt.show()

# Exemplo de uso
#grafico_silhouette_por_ponto(df_silhueta)

# %%
# FUNCTION 7
## Silhouette Suggestion Plot
#Esta função cria o gráfico da silhueta para sugestão de número de grupos
#Parâmetros:
#  df_silhueta : DataFrame obtido na função anterior contendo os resultados de silhueta para cada número de clusters.
#Retorna: Apenas exibe o gráfico da silhueta.

def grafico_silhouette_sugestao_grupos(df_silhueta):
    # Encontrar o valor máximo de s_k
    max_s_k = df_silhueta['s_k'].max()
    
    # Encontrar o menor k correspondente ao maior valor de s_k
    min_k_with_max_s_k = df_silhueta[df_silhueta['s_k'] == max_s_k]['k'].min()
    
    # Plotar o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df_silhueta['k'], df_silhueta['s_k'], marker='o', linestyle='-', color='b')
    
    # Adicionar uma linha vertical pontilhada no menor k com s_k máximo
    plt.axvline(x=min_k_with_max_s_k, color='r', linestyle='--', label=f'k = {min_k_with_max_s_k}')
    
    # Títulos e rótulos dos eixos
    plt.title('Gráfico da Silhueta')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Silhueta')
    
    # Adicionar legenda
    plt.legend()
    
    # Mostrar o gráfico
    plt.show()

# Exemplo de uso
#grafico_silhouette_sugestao_grupos(df_silhueta)


# %%
# DATA
Candidate = ['A', 'B', 'C', 'D', 'E', 'F']
GraduatedYrs = [5, 3, 8, 6, 2, 7]
LastJobYrs = [7, 4, 6, 5, 3, 9]
EducationYrs = [6, 5, 7, 8, 2, 4]

# DataFrame
df = pd.DataFrame({
    'Candidate': Candidate,
    'GraduatedYrs': GraduatedYrs,
    'LastJobYrs': LastJobYrs,
    'EducationYrs': EducationYrs
})

df.head(6)

# %%
df_adj = df.copy()

# Set 'Candidate' as index
df_adj.set_index('Candidate', inplace=True)

df_adj.head(6)

# %%
# Calculate pairwise distances
d = pdist(df_adj, metric='euclidean')
d

# %%
# better visualization of the distance matrix
matriz = squareform(d)
pd.DataFrame(matriz, index=df_adj.index, columns= df_adj.index)

# %%
# Clustering - Hierarchical Clustering
#criando clusters a partir do metodo escolhido
HC_single = linkage(d, method='single')
HC_complete = linkage(d, method='complete')
HC_single_average = linkage(d, method='average')
HC_centroid = linkage(d, method='centroid',)
HC_ward = linkage(d, method='ward')

# %%
# Plotting dendrograms for different methods

# Single Linkage
plt.figure(figsize=(5, 3))
plt.title("Single Linkage Dendogram - nearest neighbor")
dendrogram(HC_single, labels=df_adj.index)
plt.show()

# %%
# Plotting dendrograms for different methods

# Complete Linkage
plt.figure(figsize=(5, 3))
plt.title("Complete Linkage Dendogram - farthest neighbor")
dendrogram(HC_complete, labels=df_adj.index)
plt.show()

# %%
# Plotting dendrograms for different methods

# Average Linkage
plt.figure(figsize=(5, 3))
plt.title("Average Linkage Dendogram - average distance")
dendrogram(HC_single_average, labels=df_adj.index)
plt.show()

# %%
# Plotting dendrograms for different methods

# Centroid Linkage
plt.figure(figsize=(5, 3))
plt.title("Centroid Linkage Dendogram")
dendrogram(HC_centroid, labels=df_adj.index)
plt.show()

# %%
# Plotting dendrograms for different methods

# Ward Linkage
plt.figure(figsize=(5, 3))
plt.title("Ward Method Dendogram - variance minimization")
dendrogram(HC_ward, labels=df_adj.index)
plt.show()

# %%
#Comparando o método da média e ward
#Usando a opção pronta, sem label pelo tangleram
tanglegram.plot(HC_single_average, HC_ward, sort=False) 
plt.show()

# %%

# Get labels for the dendrogram
df_adj_list = df_adj.index.tolist()  
df_mapping = pd.DataFrame({'Index': range(len(df_adj_list)), 'Aluno': df_adj_list})
df_mapping

# %%
fazer_dendograma(df_adj, metodo='ward', padronizar=False)

# %%
grafico_elbow(df_adj, metodo='ward', max_clusters=10, padronizar = False)

# %%
# chamando a funcao para calcular o a_i, b_i, s_i e s_k base de dados
df_silhueta = s_silhouette(df_adj, metodo='ward', max_clusters=10, padronizar=False)
# Exibindo os resultados
#caso queira ver todas as linhas da tabela
pd.set_option('display.max_rows', None)
df_silhueta
#df_silhueta[df_silhueta['k']==5]
#df_silhueta[['s_k', 'k']].drop_duplicates()

# %%
grafico_silhouette_sugestao_grupos(df_silhueta)

# %%
grafico_silhouette_por_ponto(df_silhueta)

# %%
# Create clusters based on the dendrogram
group = fcluster(HC_ward, t=2, criterion='maxclust')
df_adj['group'] = group
df_adj

# %%
## EDA on both groups
# Group 1
group_1 = df_adj[df_adj['group'] == 1]
group_1.describe()

# %%
# Group 2
group_2 = df_adj[df_adj['group'] == 2]
group_2.describe()

# %%

vars_analise = ['GraduatedYrs', 'LastJobYrs', 'EducationYrs']

# Definir quantos gráficos por linha (exemplo: 2)
graficos_por_linha = 2
# Calcular o número de linhas necessárias
num_linhas = -(-len(vars_analise) // graficos_por_linha)  
# Criar figura de subplots
fig, axes = plt.subplots(num_linhas, graficos_por_linha, figsize=(12, 4 * num_linhas))
# Ajustar o layout dos subplots
fig.tight_layout(pad=5.0)

for i, var in enumerate(vars_analise):
    ax = axes[i // graficos_por_linha, i % graficos_por_linha]  # Seleciona a posição correta do subplot
    
    # Criar uma nova coluna 'Grupo' que mantém os clusters e adiciona 'Geral' apenas para a plotagem
    df_adj['Grupo'] = df_adj['group'].astype(str)
    
    # Criar uma cópia temporária para o grupo 'Geral' contendo todas as observações
    df_adj_geral = df_adj.copy()
    df_adj_geral['Grupo'] = 'Geral'
    
    # Concatenar os dados para garantir que 'Geral' contenha todos os dados e os clusters fiquem separados
    df_adj_combined = pd.concat([df_adj, df_adj_geral])
    
    # Definir a ordem das categorias com 'Geral' na primeira posição
    order = ['Geral'] + sorted(df_adj['Grupo'].unique().astype(str))
    
    # Plotar o boxplot geral e por cluster no subplot
    sns.boxplot(ax=ax, x='Grupo', y=var, data=df_adj_combined, palette='Set1', order=order)
    
    # Ajustar o título e rótulos
    ax.set_title(f'Boxplots de {var}: Geral e por Cluster')
    ax.set_xlabel('Grupo')
    ax.set_ylabel(var)

plt.show()


# %%
df_adj

# %%
df_adj_geral

# %%
