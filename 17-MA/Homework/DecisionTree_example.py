# %% 
# LIBRARY
import os
import pandas as pd
from sklearn.tree import plot_tree, DecisionTreeClassifier #importando as funções de visualizar a árvore e criar a árvore
from matplotlib import pyplot as plt # importando a biblioteca de visualização de dados 

# %%
project_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Project_Code", "ASN-DSA-T5", "17-MA")

# Data path
data_path = os.path.join(project_dir, "data")
os.makedirs(data_path, exist_ok=True)

# %%
# Load data
base_celulas = pd.read_excel(os.path.join(data_path, "celulas.xlsx"))
base_celulas


# %%
#No python é comum definir objetos diferentes sendo o Y e os x's.
#muda a forma de olhar, pois vamos ter um objeto que representa apenas a respota, e outro que são as variáveis de suporte
#e então o modelo é criado conforme a "carcaça", o "molde".

#para definir o que são as variáveis X eu só deletei a variável Classe
X = base_celulas.drop(["Classe"], axis = 1)
X.head()


# %%
#para definir o Y selecionei apenas a coluna Classe
y = base_celulas["Classe"]
y.head()

# %%

#Na árvore de decisão do python você precisa criar as dummies das variáveis X para colocar na árvore
#aqui estou utilizando a get_dummies do pandas. Deixarei uma nota abaixo com a explicação do porquê 
#esta função e não a OneHotEnconder (já explico as diferenças)

#criando dummie das variáveis X's categóricas, utilizando a função drop_first
X = pd.get_dummies(X, columns=['Cor', 'Membrana'], drop_first=True)
X

# %%
#criando a carcaça do modelo
#repare que aqui é só a definição do modelo, olha ali os hiper parâmetros:
celulas_tree_model = DecisionTreeClassifier(random_state = 42, #semente para aleatórios
                                  min_samples_split = 2,  #tamanho min do nó a ser dividido
                                  ccp_alpha = 0,  #cost complexity
                                  max_depth = 5,  # profundidade máxima
                                  criterion='gini') #método poderia ser: entropy

#agora sim estou mandando executar o modelo
celulas_tree_model.fit(X, y) # treinamento do modelo


# %%
print(celulas_tree_model.classes_)


# %%
#pedindo o desenho da árvore de decisão
plt.figure(figsize=(15, 15)) # aumentanto o tamanho do gráfico
plot_tree(celulas_tree_model, # especificando o modelo
          feature_names=X.columns, # especificando o nome das variáveis que estamos usando para a classificação
          class_names= celulas_tree_model.classes_, #adicionando nome das classes
          #para saber os valores do y, você pode utilizar o comando print(celulas_tree_model.classes_)
          filled=True) # em cada nó, irá mostrar uma cor relacionada a cada categoria mais frequente da variável resposta
plt.show() # comando para visualizar a árvore


# %%
# carregando os dados
dados_compra = pd.read_excel(os.path.join(data_path, "compra.xlsx"))
dados_compra

# %%
#definindo o que é X e o que é Y
X_compra = dados_compra.drop(['Compra'],axis = 1)
y_compra = dados_compra["Compra"]

# %%
#criando dummie das variáveis categóricas
X_compra= pd.get_dummies(X_compra, columns=['Sexo',"Pais"])
X_compra

# %%
compra_tree_model = DecisionTreeClassifier(random_state = 42,
                                  min_samples_split = 2, 
                                  ccp_alpha = 0, 
                                  max_depth = 5, 
                                  criterion='entropy') # trocando o critério para entropia
compra_tree_model.fit(X_compra, y_compra)

# %%
print(compra_tree_model.classes_)

# %%
plt.figure(figsize=(15, 15))
plot_tree(compra_tree_model,
           feature_names=X_compra.columns, # especificando o nome das variáveis que estamos usando para a classificação
          class_names= compra_tree_model.classes_,  # especificando o nome das categorias da variável resposta 
          filled=True)
plt.show()

# %%
