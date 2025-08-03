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
library(readxl)
library(NbClust) #ajuda na silhueta
library(GGally) #grafico de coordenadas paralelas

########################################
#
#     CLUSTER HIERARQUICO - MCDonald
#
########################################

#Carregar base de dados: 
mcdonalds <- read.table("C:/Users/kel_m/OneDrive/Studies/2024_ASN-DSAT5/23_AC/Class/Codigos_R_Python/dados/MCDONALDS.csv", sep = ";", dec = ",", header = T)
#transformar o nome dos lanches em linhas
rownames(mcdonalds) <- mcdonalds[,1]
mcdonalds <- mcdonalds[,-1]

#Padronizar variaveis
mcdonalds_padronizado <- scale(mcdonalds)

#calcular as distancias da matriz utilizando a distancia euclidiana
distancia <- dist(mcdonalds_padronizado, method = "euclidean")

#Calcular o Cluster: metodos disponiveis "average", "single", "complete" e "ward.D"
ch_single <- hclust(distancia, method = "single" )
ch_complete <- hclust(distancia, method = "complete" )
ch_average <- hclust(distancia, method = "average" )
ch_centroid <- hclust(distancia, method = "centroid" )
ch_ward <- hclust(distancia, method = "ward.D" )

# Dendrograma + elbow + silhouete para fechar numero de grupos

#metodo single
#elbow
elb_single <- fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "wss", hc_method = "single")
elb_single
#pensaria em 2, 5 ou 8 
#dendograma
plot(ch_single, cex = 0.6, hang = -1)
rect.hclust(ch_single, k = 2,  border = "orange")
rect.hclust(ch_single, k = 5,  border = "pink")
rect.hclust(ch_single, k = 8,  border = "blue")
#silhouete
fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "silhouette", hc_method = "single")
#pensaria em 2, 3 ou 8 
#pensando em 2 grupos ver como fica o s
grupo_single_2 <- cutree(ch_single, k = 2)
s_values_single_2 <- silhouette(grupo_single_2, distancia)
rownames(s_values_single_2) <- rownames(mcdonalds)
fviz_silhouette(s_values_single_2, label = TRUE)
#pensando em 3 grupos ver como fica o s
grupo_single_3 <- cutree(ch_single, k = 3)
s_values_single_3 <- silhouette(grupo_single_3, distancia)
rownames(s_values_single_3) <- rownames(mcdonalds)
fviz_silhouette(s_values_single_3, label = TRUE)
#pensando em 5 grupos ver como fica o s
grupo_single_5 <- cutree(ch_single, k = 5)
s_values_single_5 <- silhouette(grupo_single_5, distancia)
rownames(s_values_single_5) <- rownames(mcdonalds)
fviz_silhouette(s_values_single_5, label = TRUE)
#pensando em 8 grupos ver como fica o s
grupo_single_8 <- cutree(ch_single, k = 8)
s_values_single_8 <- silhouette(grupo_single_8, distancia)
rownames(s_values_single_8) <- rownames(mcdonalds)
fviz_silhouette(s_values_single_8, label = TRUE)


#metodo complete
#elbow
elb_complete <- fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "wss", hc_method = "complete")
elb_complete
#pensaria em 4
#dendograma
plot(ch_complete, cex = 0.6, hang = -1)
rect.hclust(ch_complete, k = 4,  border = "blue")
#silhouete
fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "silhouette", hc_method = "complete")
#pensaria em 2 ou 4 
#pensando em 2 grupos ver como fica o s
grupo_complete_2 <- cutree(ch_complete, k = 2)
s_values_complete_2 <- silhouette(grupo_complete_2, distancia)
rownames(s_values_complete_2) <- rownames(mcdonalds)
fviz_silhouette(s_values_complete_2, label = TRUE)
#pensando em 4 grupos ver como fica o s
grupo_complete_4 <- cutree(ch_complete, k = 4)
s_values_complete_4 <- silhouette(grupo_complete_4, distancia)
rownames(s_values_complete_4) <- rownames(mcdonalds)
fviz_silhouette(s_values_complete_4, label = TRUE)



#metodo average
#elbow
elb_average <- fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "wss", hc_method = "average")
elb_average
#pensaria em 4 ou 7 
#dendograma
plot(ch_average, cex = 0.6, hang = -1)
rect.hclust(ch_average, k = 4,  border = "pink")
rect.hclust(ch_average, k = 7,  border = "blue")
#silhouete
fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "silhouette", hc_method = "average")
#pensaria em 2 ou 5 
#pensando em 2 grupos ver como fica o s
grupo_average_2 <- cutree(ch_average, k = 2)
s_values_average_2 <- silhouette(grupo_average_2, distancia)
rownames(s_values_average_2) <- rownames(mcdonalds)
fviz_silhouette(s_values_average_2, label = TRUE)
#pensando em 4 grupos ver como fica o s
grupo_average_4 <- cutree(ch_average, k = 4)
s_values_average_4 <- silhouette(grupo_average_4, distancia)
rownames(s_values_average_4) <- rownames(mcdonalds)
fviz_silhouette(s_values_average_4, label = TRUE)
#pensando em 5 grupos ver como fica o s
grupo_average_5 <- cutree(ch_average, k = 5)
s_values_average_5 <- silhouette(grupo_average_5, distancia)
rownames(s_values_average_5) <- rownames(mcdonalds)
fviz_silhouette(s_values_average_5, label = TRUE)
#pensando em 7 grupos ver como fica o s
grupo_average_7 <- cutree(ch_average, k = 7)
s_values_average_7 <- silhouette(grupo_average_7, distancia)
rownames(s_values_average_7) <- rownames(mcdonalds)
fviz_silhouette(s_values_average_7, label = TRUE)
#esse ficou todo mundo positivo, pelo menos
#menores distancias dentro do seu grupo



#metodo centroid
#elbow
elb_centroid <- fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "wss", hc_method = "centroid")
elb_centroid
#pensaria em 4, 6 ou 8 
#dendograma
plot(ch_centroid, cex = 0.6, hang = -1)
rect.hclust(ch_centroid, k = 4,  border = "pink")
rect.hclust(ch_centroid, k = 6,  border = "orange")
rect.hclust(ch_centroid, k = 8,  border = "blue")
#silhouete
fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "silhouette", hc_method = "centroid")
#pensaria em 2 ou 8 
#pensando em 2 grupos ver como fica o s
grupo_centroid_2 <- cutree(ch_centroid, k = 2)
s_values_centroid_2 <- silhouette(grupo_centroid_2, distancia)
rownames(s_values_centroid_2) <- rownames(mcdonalds)
fviz_silhouette(s_values_centroid_2, label = TRUE)
#pensando em 4 grupos ver como fica o s
grupo_centroid_4 <- cutree(ch_centroid, k = 4)
s_values_centroid_4 <- silhouette(grupo_centroid_4, distancia)
rownames(s_values_centroid_4) <- rownames(mcdonalds)
fviz_silhouette(s_values_centroid_4, label = TRUE)
#pensando em 6 grupos ver como fica o s
grupo_centroid_6 <- cutree(ch_centroid, k = 6)
s_values_centroid_6 <- silhouette(grupo_centroid_6, distancia)
rownames(s_values_centroid_6) <- rownames(mcdonalds)
fviz_silhouette(s_values_centroid_6, label = TRUE)
#pensando em 8 grupos ver como fica o s
grupo_centroid_8 <- cutree(ch_centroid, k = 8)
s_values_centroid_8 <- silhouette(grupo_centroid_8, distancia)
rownames(s_values_centroid_8) <- rownames(mcdonalds)
fviz_silhouette(s_values_centroid_8, label = TRUE)



#metodo ward
#elbow
elb_ward <- fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "wss", hc_method = "ward.D")
elb_ward
#pensaria em 4 ou 5 
#dendograma
plot(ch_ward, cex = 0.6, hang = -1)
rect.hclust(ch_ward, k = 4,  border = "blue")
rect.hclust(ch_ward, k = 5,  border = "orange")
#silhouete
fviz_nbclust(mcdonalds_padronizado, FUN = hcut, method = "silhouette", hc_method = "ward.D")
#pensaria em 4 ou 7 
#pensando em 4 grupos ver como fica o s
grupo_ward_4 <- cutree(ch_ward, k = 4)
s_values_ward_4 <- silhouette(grupo_ward_4, distancia)
rownames(s_values_ward_4) <- rownames(mcdonalds)
fviz_silhouette(s_values_ward_4, label = TRUE)
#pensando em 5 grupos ver como fica o s
grupo_ward_5 <- cutree(ch_ward, k = 5)
s_values_ward_5 <- silhouette(grupo_ward_5, distancia)
rownames(s_values_ward_5) <- rownames(mcdonalds)
fviz_silhouette(s_values_ward_5, label = TRUE)
#pensando em 7 grupos ver como fica o s
grupo_ward_7 <- cutree(ch_ward, k = 7)
s_values_ward_7 <- silhouette(grupo_ward_7, distancia)
rownames(s_values_ward_7) <- rownames(mcdonalds)
fviz_silhouette(s_values_ward_7, label = TRUE)


########################################
#
#     CLUSTER NAO HIERARQUICO - MCDonald
#
########################################
set.seed(1987)
elb_kmeans <- fviz_nbclust(mcdonalds_padronizado, kmeans, method = "wss") #quem sabe 4 grupos
elb_kmeans
fviz_nbclust(mcdonalds_padronizado, kmeans, method = "silhouette") #2, 4 ou 6?

#criando cluster kmeans
mcdonalds_k2 <- kmeans(mcdonalds_padronizado, centers = 2)
mcdonalds_k4 <- kmeans(mcdonalds_padronizado, centers = 4)
mcdonalds_k5 <- kmeans(mcdonalds_padronizado, centers = 5)
mcdonalds_k6 <- kmeans(mcdonalds_padronizado, centers = 6)

#Criar graficos
G1 <- fviz_cluster(mcdonalds_k2, geom = "point", data = mcdonalds_padronizado) + ggtitle("k = 2")
G2 <- fviz_cluster(mcdonalds_k4, geom = "point",  data = mcdonalds_padronizado) + ggtitle("k = 4")
G3 <- fviz_cluster(mcdonalds_k5, geom = "point",  data = mcdonalds_padronizado) + ggtitle("k = 5")
G4 <- fviz_cluster(mcdonalds_k6, geom = "point",  data = mcdonalds_padronizado) + ggtitle("k = 6")

grid.arrange(G1, G2, G3, G4, nrow = 2)

#observando a silhueta para cada ponto

#2 grupos
s_values_k2 <- silhouette(mcdonalds_k2$cluster, distancia)
rownames(s_values_k2) <- rownames(mcdonalds_padronizado)
fviz_silhouette(s_values_k2, label = TRUE)
#4 grupos
s_values_k4 <- silhouette(mcdonalds_k4$cluster, distancia)
rownames(s_values_k4) <- rownames(mcdonalds_padronizado)
fviz_silhouette(s_values_k4, label = TRUE)
#5 grupos
s_values_k5 <- silhouette(mcdonalds_k5$cluster, distancia)
rownames(s_values_k5) <- rownames(mcdonalds_padronizado)
fviz_silhouette(s_values_k5, label = TRUE)
#6 grupos
s_values_k6 <- silhouette(mcdonalds_k6$cluster, distancia)
rownames(s_values_k6) <- rownames(mcdonalds_padronizado)
fviz_silhouette(s_values_k6, label = TRUE)

#pelo k-means, 4 grupos, todos os individuos tem s maior que 0
#parece o mais interessante

########################################
#
#     DISCUSSAO
#
########################################

#por curiosidade eu vou ver a variabilidade de 4 grupos pelo hierarquico
#e pelo nao hierarquico

tabela_variabilidades <- data.frame(
  Clusters = elb_complete$data$clusters, 
  Single = elb_single$data$y,
  Complete = elb_complete$data$y,
  Average = elb_average$data$y,
  Centroid = elb_centroid$data$y,
  Ward = elb_ward$data$y,
  Kmeans = elb_kmeans$data$y
)
tabela_variabilidades

########################################
#
#     DECISAO ou quase - MCDonald
#
########################################
#ficaria com 4 grupos pelo kmeans
#agora eh tentar interpretar e definir um sentido para descrever cada grupo


#com a base original vou adicionar a variavel cluster gerada no kmeans 4 grupos

Base_lanches_fim <- mcdonalds %>%
  mutate(grupo_lanches4 = mcdonalds_k4$cluster)

#FAZENDO ANALISE DESCRITIVA
#MEDIAS das variaveis por grupo
mediagrupo <- Base_lanches_fim %>%
  group_by(grupo_lanches4) %>%
  summarise(n=n(), across(everything(), mean))
mediagrupo

# Criando uma copia da base mudando a variavel cluster para "Geral"
#o objetivo isso eh poder criar um boxplot da base toda e por grupo
Base_lanches_fim_geral <- Base_lanches_fim %>%
  mutate(grupo_lanches4 = "Geral")

# Convertendo a variavel 'grupo_lanches4' para 'character' antes de apendar as bases
Base_lanches_fim$grupo_lanches4 <- as.character(Base_lanches_fim$grupo_lanches4)

# Apendando os dados
Base_lanches_combined <- bind_rows(Base_lanches_fim, Base_lanches_fim_geral)

# Definindo o fator para os grupos, incluindo "Geral"
Base_lanches_combined$grupo_lanches4 <- factor(Base_lanches_combined$grupo_lanches4, 
                                               levels = c("Geral", "1", "2", "3", "4"))

# Selecionando as variaveis numericas para os boxplots
numeric_vars <- colnames(mcdonalds)  

# Criando uma lista para armazenar os graficos de boxplot
plot_list <- list()

# Loop para criar boxplots para cada variavel numerica
for (var in numeric_vars) {
  p <- ggplot(Base_lanches_combined, aes_string(x = "grupo_lanches4", y = var, fill = "grupo_lanches4")) +
    geom_boxplot() +
    labs(title = paste("Boxplot de", var, ": Geral e por Cluster"), x = "Grupo", y = var) +
    theme_minimal()
  
  plot_list[[var]] <- p  
}

# Exibir os graficos de boxplot em um grid
grid.arrange(grobs = plot_list, ncol = 2)


# Exibir os graficos de boxplot em um grid com 4 graficos por pagina 
# Função para plotar em "batches" de 4 gráficos por vez
num_plots <- length(plot_list)
# Loop para exibir 4 graficos por pagina
for (i in seq(1, num_plots, by = 4)) {
  # Definindo o número de gráficos a ser exibido na iteração atual
  grid.arrange(grobs = plot_list[i:min(i+3, num_plots)], ncol = 2, nrow = min(2, ceiling((num_plots-i+1)/2)))
}


