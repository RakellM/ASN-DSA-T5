#-items in work space
ls()

#-remove all
rm(list=ls())

# LIBRARY
## Packages
load.pks <- c(
  "readr",
  "readxl", # Read file
  "e1071", # Calculate Skewness and Kutosis
  #"lubridate",
  "scales", # Standardize variables
  "dplyr",
  #"tidyr",
  "tidyverse", # Data manipulation
  "Hmisc", # Summary for categorical variables
  "DescTools", # Mode calculation
  #"esquisse",
  "kableExtra",
  #"ClusterR",
  "cluster", # Cluster algorithm
  "dendextend", #compare dendograms
  "factoextra", # Cluster algorithm and Visualization
  "fpc", # Cluster algorithm and Visualization
  "gridExtra", # Grid arrange fucntion
  #"pivottabler",
  #"ggcorrplot",
  #"corrplot",
  #"MASS",
  #"RMySQL",
  #"DBI",
  #"htmlwidgets",
  "knitr",
  #"bigrquery",
  "PerformanceAnalytics", # Correlation Chart
  "GGally",    # Correlation charts
  "patchwork", # Combine plots
  "plotly",
  "ggplot2"
)

#-- Install
is_installed <- load.pks %in% rownames(installed.packages())
if(any(is_installed == FALSE)){
  install.packages(load.pks[!is_installed])
}
#-- Load packages
sapply(load.pks, require, character.only = TRUE)

# DATA
data0 <- read_csv2("C:/Users/kel_m/OneDrive/Studies/2024_ASN-DSAT5/23_AC/Homework/Lista_02/municipios.csv",
                   locale = locale(encoding = "latin1", decimal_mark = ","))

head(data0)

#-- View structure
str(data0)

#-- Column names
names(data0)

#-- Dataset dimensions
dim(data0)

## Filters
#-- It was advised to remove "São Paulo" & "Campinas"
data0["Município"]
data1a <- data0 %>%
  filter(!Município %in% c("São Paulo", "Campinas")) %>%
  filter(!Esgoto_2010 < 80)

# EDA - Exploration Data Analysis
#-- Basic summary
summary(data1a)

#-- For categorical variables
describe(data1a)

## New Variables
### Active Population
data1a <- data1a %>%
  mutate(
    Pop_Ativa = 100 - (Pop_Menos_15Anos_2014 + Pop_com_60Anoso_2014),
    Hab_Ativo = (Pop_Ativa / 100) * Habitantes ,
    Emprego_pct = (Empregos_formais_2013 / Hab_Ativo) * 100 ,
    MaoObra_disponivel = case_when(Emprego_pct < 100 ~ 100 - Emprego_pct,
                                   Emprego_pct >= 100 ~ 0)
  )


data1a %>%
  select(Município, Emprego_pct, Habitantes, Hab_Ativo, Empregos_formais_2013) %>%
  filter(Emprego_pct > 80) %>%
  arrange(desc(Emprego_pct))

data1b <- as.data.frame(data1a)
rownames(data1b) <- data1b[,1]
data1b <- data1b[,-1]

data1c <- data1b %>%
  filter(Emprego_pct <= 80) %>%
  filter(Hab_Ativo >= 5000)

data1d <- scale(data1c)

data1 <- as.data.frame(data1d) 
 
## Each numeric variable
data_sum <- data1c
var_names <- colnames(data_sum)
var_names

#-- Change index [i]  to get each variable summary table and chart
i <- 12
### Summary Table
data_sum_Var2 <- data_sum %>%
  select(var_names[i]) %>%
  reframe(
    qnty = n(),
    missing = qnty - sum(!is.na( .data[[var_names[i]]] )) ,
    min = round( min( .data[[var_names[i]]] , na.rm=TRUE) , 1),
    max = round( max( .data[[var_names[i]]] , na.rm=TRUE) , 1),
    mean = round( mean( .data[[var_names[i]]] , na.rm=TRUE) , 1),
    median = round( median( .data[[var_names[i]]], na.rm=TRUE) , 1),
    mode = round( Mode( .data[[var_names[i]]] , na.rm=TRUE) , 1),
    freq_mode = sum( .data[[var_names[i]]] %in% 
                      Mode( .data[[var_names[i]]] , na.rm = TRUE), 
                    na.rm = TRUE),  # Counts occurrences of all modes
    std_dev = round( sd( .data[[var_names[i]]] , na.rm=TRUE) , 1),
    variance = round( sd( .data[[var_names[i]]] , na.rm=TRUE)^2 , 1),
    cv = round( sd( .data[[var_names[i]]] , na.rm=TRUE) / mean * 100 , 1),
    skewness = round( skewness( .data[[var_names[i]]] , na.rm=TRUE) , 1),
    curtose = round( kurtosis( .data[[var_names[i]]] , na.rm=TRUE), 1)
  )

data_sum_Var2 %>%  
  kbl(caption = paste("Summary Statistics for", var_names[i])) %>% 
  kable_paper("hover", full_width = F)


### Histogram and Density
g_hist = ggplot(data_sum, aes(x = .data[[var_names[i]]] )) +
  geom_histogram(color = "black",
                 fill = "lightblue",
                 bins = 10,
                 aes(y = (..count..) / sum(..count..))) +
  geom_density(col = 2, linewidth = 1, aes(y = 30 * (..count..) /  sum(..count..))) +
  ggtitle("Histogram & Density") +
  xlab(var_names[i]) +
  ylab("Relative Frequency")

### BoxPlot
g_boxplot = ggplot(data_sum, aes(x = .data[[var_names[i]]] )) +
  geom_boxplot(
    fill = "lightblue",           # Box color
    color = "black",           # Border color
    outlier.color = "red",        # Outliers in red
    outlier.shape = 19            # Outliers as solid dots
  ) +
  # Add mean as red "X" using annotate() (avoids data length mismatch)
  annotate(
    geom = "point",
    y = 0,
    x = round( mean( data_sum[[var_names[i]]] , na.rm=TRUE) , 1),
    shape = 4,
    size = 3,                    
    color = "red",
    stroke = 1.5
  ) +
  labs(title = "Boxplot", y = "", x = var_names[i])

### Group Charts
grid.arrange(g_hist,
             g_boxplot,
             nrow = 2,
             ncol = 1)



# PRE MODELING
data3 <- data1 %>%
  select(Esgoto_2010, 
         taxa_natalidade,
         #Emprego_pct,
         area,
         PIB,
         MaoObra_disponivel,
         #Habitantes,
         #Pop_Menos_15Anos_2014,
         #Pop_com_60Anoso_2014,
         #Hab_Ativo,
         Pop_Ativa) 
## Correlation Matrix
cor_matrix <- cor(data3, use = "complete.obs", method = "pearson")
print(round(cor_matrix, 3))

chart.Correlation(data3, histogram = TRUE)

# MODELING
## Hierarquical Cluster
#-- Distance Matrix
d2 <- dist(data3, method = "euclidean")
d2

#-- Define the cluster method
hc_1 <- hclust(d2, method = "single" )
hc_2 <- hclust(d2, method = "complete" )
hc_3 <- hclust(d2, method = "average" )
hc_4 <- hclust(d2, method = "ward.D" )
hc_5 <- hclust(d2, method = "centroid" )

# Configurar o grid (3 linhas, 2 colunas)
par(mfrow = c(3, 2), mar = c(2, 4, 1, 1))  # margens menores para economizar espaço

#-- Dendogram
plot(hc_1, cex = 0.6, hang = -1, main = "Dendogam - Single Linkage")
plot(hc_2, cex = 0.6, hang = -1, main = "Dendogam - Complete Linkage")
plot(hc_3, cex = 0.6, hang = -1, main = "Dendogam - Average Linkage")
plot(hc_4, cex = 0.6, hang = -1, main = "Dendogam - Ward Method")
plot(hc_5, cex = 0.6, hang = -1, main = "Dendogam - Centroid")
plot.new()

### Elbow
#-- Group Suggestion
elbow1 <- fviz_nbclust(data3, FUN = hcut, method = "wss", hc_method = "single") +
  ggtitle("Method: Single Linkage")
elbow2 <- fviz_nbclust(data3, FUN = hcut, method = "wss", hc_method = "complete") +
  ggtitle("Method: Complete Linkage")
elbow3 <- fviz_nbclust(data3, FUN = hcut, method = "wss", hc_method = "average") +
  ggtitle("Method: Average Linkage")
elbow4 <- fviz_nbclust(data3, FUN = hcut, method = "wss", hc_method = "ward.D") +
  ggtitle("Method: Ward")
elbow5 <- fviz_nbclust(data3, FUN = hcut, method = "wss", hc_method = "centroid") +
  ggtitle("Method: Centroid")

grid.arrange(elbow1, elbow2, elbow3, elbow4, elbow5 ,
             nrow = 3, ncol = 2)


#-- Elbow chart values
wss_values <- elbow1$data
wss_values

### Sillhouete
#-- Group Suggestion
sil1 <- fviz_nbclust(data3, FUN = hcut, method = "silhouette", hc_method = "single") +
  ggtitle("Method: Single Linkage")
sil2 <- fviz_nbclust(data3, FUN = hcut, method = "silhouette", hc_method = "complete") +
  ggtitle("Method: Complete Linkage")
sil3 <- fviz_nbclust(data3, FUN = hcut, method = "silhouette", hc_method = "average") +
  ggtitle("Method: Average Linkage")
sil4 <- fviz_nbclust(data3, FUN = hcut, method = "silhouette", hc_method = "ward.D") +
  ggtitle("Method: Ward")
sil5 <- fviz_nbclust(data3, FUN = hcut, method = "silhouette", hc_method = "centroid") +
  ggtitle("Method: Centroid")

grid.arrange(sil1, sil2, sil3, sil4, sil5 ,
             nrow = 3, ncol = 2)


## Choose
par(mfrow = c(1, 1), mar = c(2, 4, 1, 1)) 
hc_chosen <- hc_4
nbr_group <- 6
plot(hc_chosen, cex = 0.6, hang = -1)
rect.hclust(hc_chosen, k = nbr_group)

#-- Groups
groups <- cutree(hc_chosen, k = nbr_group)

#-- Silhouete s values
silhouette_values <- silhouette(groups, d2)
rownames(silhouette_values) <- rownames(data3)
#verificando os valores de cada ponto
silhouette_values
#grafico da silhueta
fviz_silhouette(silhouette_values, label = TRUE)


## Analise per Group
#data4 <- data.frame(Município = data1$Município, data3, cluster = groups)
data4 <- data.frame(data1c, cluster = groups)

# Função para plotar boxplot dinâmico
plot_boxplot_dinamico <- function(dados, variavel, coluna_cluster = "cluster") {
  
  # Garantir que a coluna de cluster seja fator
  dados[[coluna_cluster]] <- as.factor(dados[[coluna_cluster]])
  
  # Níveis dos clusters (ex: "1", "2", "3", ...)
  niveis_clusters <- levels(dados[[coluna_cluster]])
  
  # Criar dataframe combinando Geral + Clusters
  dados_plot <- dados %>%
    select({{variavel}}, all_of(coluna_cluster)) %>%
    mutate(tipo = "All") %>%
    bind_rows(
      dados %>%
        select({{variavel}}, all_of(coluna_cluster)) %>%
        mutate(tipo = as.character(.[[coluna_cluster]]))
    ) %>%
    mutate(tipo = factor(tipo, levels = c("All", niveis_clusters)))
  
  # Cores (Geral = cinza, clusters = cores automáticas)
  cores <- c("All" = "gray70", scales::hue_pal()(length(niveis_clusters)))
  names(cores)[-1] <- niveis_clusters  # Nomeia as cores conforme os clusters
  
  # Plot
  ggplot(dados_plot, aes(x = tipo, y = .data[[variavel]], fill = tipo)) +
    geom_boxplot(alpha = 0.7, width = 0.6) +
    scale_fill_manual(values = cores) +
    labs(
      title = paste("Distribuition of", variavel, ": All vs Clusters"),
      x = "",
      y = variavel,
      fill = "Cluster"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
}

# Exemplo de uso:
p1 <- plot_boxplot_dinamico(data4, "Pop_Ativa")
p2 <- plot_boxplot_dinamico(data4, "Esgoto_2010")
p3 <- plot_boxplot_dinamico(data4, "taxa_natalidade")
p4 <- plot_boxplot_dinamico(data4, "area")
p5 <- plot_boxplot_dinamico(data4, "Empregos_formais_2013")
p6 <- plot_boxplot_dinamico(data4, "Emprego_pct")
p7 <- plot_boxplot_dinamico(data4, "PIB")
p8 <- plot_boxplot_dinamico(data4, "Pop_Menos_15Anos_2014")
p9 <- plot_boxplot_dinamico(data4, "Pop_com_60Anoso_2014")
p10 <- plot_boxplot_dinamico(data4, "Habitantes")
p11 <- plot_boxplot_dinamico(data4, "MaoObra_disponivel")

(p1 + p2) / (p3 + p4) / (p11 + p7) / (p8 + p9)
p1 + p2 + p11

## 1. Create Combined Summary Statistics (Fixed)
combined_stats <- data4 %>%
  #select(-Município) %>%
  # Calculate overall statistics
  mutate(Cluster = "Overall") %>%
  bind_rows(
    data4 %>%
      #select(-Município) %>%
      mutate(Cluster = as.character(cluster))
  ) %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric),
                   list(
                     mean = ~mean(., na.rm = TRUE),
                     sd = ~sd(., na.rm = TRUE),
                     min = ~min(., na.rm = TRUE),
                     median = ~median(., na.rm = TRUE),
                     max = ~max(., na.rm = TRUE),
                     n = ~n()
                   ),
                   .names = "{.col}@{.fn}"  # Using @ instead of _ to avoid conflict
  )) %>%
  pivot_longer(
    -Cluster,
    names_to = c("Variable", ".value"),
    names_sep = "@"  # Changed separator
  ) %>%
  mutate(across(where(is.numeric), ~round(., 2)))

combined_stats %>%  
  kbl %>% 
  kable_paper("hover", full_width = F)

data4 %>%
  filter(cluster == 1) %>%
  kbl %>% 
  kable_paper("hover", full_width = F)

