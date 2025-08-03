########################################
#
#   CHAMANDO BIBLIOTECAS IMPORTANTES
#
########################################

library(tidyverse) #pacote para manipulacao de dados
library(cluster) #algoritmo de cluster
library(dendextend) #compara dendogramas
library(factoextra) #algoritmo de cluster e visualizacao
library(fpc) #algoritmo de cluster e visualizacao
library(gridExtra) #para a funcao grid arrange
library(readxl) #carregar dados
library(NbClust) #ajuda na silhueta
library(GGally) #grafico de coordenadas paralelas


########################################
#
#    CLUSTER NAO HIERARQUICO - Clientes
#
########################################

#carregar base
#Descricao das variaveis (dicionario de dados):
#Channel: O canal de vendas utilizado pelo cliente. (1: Horeca (Hotel/Restaurante/Café) / 2: Varejista.)
#Region: A região onde o cliente está localizado. (1: Lisboa. / 2: Porto. / 3: Outras regiões.)
#Fresh: Gastos anuais do cliente com produtos frescos (frutas, vegetais, etc.) em unidades monetarias (por exemplo, euros).
#Milk: Gastos anuais do cliente com produtos lacteos (leite, queijos, etc.).
#Grocery: Gastos anuais do cliente com mantimentos (alimentos embalados, enlatados, etc.).
#Frozen: Gastos anuais do cliente com produtos congelados.
#Detergents_Paper: Gastos anuais do cliente com produtos de limpeza e papel.
#Delicassen: Gastos anuais do cliente com itens de delicatessen (produtos finos como carnes frias, queijos especiais, etc.).


wholesale_customers <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv")
head(wholesale_customers)

#retirando as categoricas para rodar o cluster
wholesale_customers_num <- wholesale_customers %>% 
  select(-Region, -Channel)

#padronizar dados
padronizado <- scale(wholesale_customers_num)

#verificando Elbow
set.seed(733)
elbow_kmeans <- fviz_nbclust(padronizado, FUN = kmeans, method = "wss")
elbow_kmeans 

#verificando silhueta
fviz_nbclust(padronizado, FUN = kmeans, method = "silhouette")

#Agora vamos rodar de 3 a 6 centros  e visualizar qual a melhor divisao
k2 <- kmeans(padronizado, centers = 2)
k3 <- kmeans(padronizado, centers = 3)
k4 <- kmeans(padronizado, centers = 4)
k5 <- kmeans(padronizado, centers = 5)

#Graficos
G1 <- fviz_cluster(k2, geom = "point", data = padronizado) + ggtitle("k = 2")
G2 <- fviz_cluster(k3, geom = "point",  data = padronizado) + ggtitle("k = 3")
G3 <- fviz_cluster(k4, geom = "point",  data = padronizado) + ggtitle("k = 4")
G4 <- fviz_cluster(k5, geom = "point",  data = padronizado) + ggtitle("k = 5")

#Criar uma matriz com 4 graficos
grid.arrange(G1, G2, G3, G4, nrow = 2)


#por curiosidade estou vendo o s de cada ponto
silhouette_values_k2 <- silhouette(k2$cluster, dist(padronizado, method = "euclidean"))
fviz_silhouette(silhouette_values_k2, label = TRUE)

silhouette_values_k3 <- silhouette(k3$cluster, dist(padronizado, method = "euclidean"))
fviz_silhouette(silhouette_values_k3, label = TRUE)

silhouette_values_k4 <- silhouette(k4$cluster, dist(padronizado, method = "euclidean"))
fviz_silhouette(silhouette_values_k4, label = TRUE)

silhouette_values_k5 <- silhouette(k5$cluster, dist(padronizado, method = "euclidean"))
fviz_silhouette(silhouette_values_k5, label = TRUE)


#vou analisar cinco clusters
base_cluster <- wholesale_customers %>%
  mutate(grupo = k5$cluster)

mediagrupo <- base_cluster %>%
  select(-Region, -Channel) %>% 
  group_by(grupo) %>%
  summarise(n=n(), across(everything(), mean))
mediagrupo

#quero ver o boxplot geral e por cada um dos grupos

# criando uma copia da base mudando a variavel cluster para Geral
base_cluster_geral <- base_cluster %>%
  mutate(grupo = "Geral")

# combinando os dados do grupo 'Geral' com os clusters criados (para poder plotar tudo junto)
base_cluster$grupo <- as.character(base_cluster$grupo)
base_cluster_combined <- bind_rows(base_cluster, base_cluster_geral)
base_cluster_combined$grupo <- factor(base_cluster_combined$grupo, 
                                      levels = c("Geral", "1", "2", "3", "4", "5"))


# selecionando somente as variaveis numericas para os boxplots
numeric_vars <- c("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")

# loop para criar boxplots para cada variavel numerica
plot_list <- list()
for (var in numeric_vars) {
  p <- ggplot(base_cluster_combined, aes_string(x = "grupo", y = var, fill = "grupo")) +
    geom_boxplot() +
    labs(title = paste("Boxplot de", var, ": Geral e por Cluster"), x = "Grupo", y = var) +
    theme_minimal()
  
  plot_list[[var]] <- p  
}
grid.arrange(grobs = plot_list, ncol = 2)


#quero ver o grafico de barras geral e por cada um dos grupos
# transformando as variaveis categoricas em fatores para poder colocar no fill do grafico
base_cluster_combined$Channel <- as.factor(base_cluster_combined$Channel)
base_cluster_combined$Region <- as.factor(base_cluster_combined$Region)

#selecionando as variaveis categoricas
categorical_vars <- c("Channel", "Region")

# loop para criar graficos de barras para cada variavel categorica
plot_list_cat <- list()
for (var in categorical_vars) {
  p <- ggplot(base_cluster_combined, aes(x = grupo, fill = .data[[var]])) +
    geom_bar(position = "dodge") +
    labs(title = paste("Contagem de", var, ": Geral e por Cluster"), x = "Grupo", y = "Contagem") +
    theme_minimal() +
    scale_fill_brewer(palette = "Set1")  
  plot_list_cat[[var]] <- p  
}

grid.arrange(grobs = plot_list_cat, ncol = 2)
