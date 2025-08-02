#-items in work space
ls()

#-remove all
rm(list=ls())

# LIBRARY
## Packages
load.pks <- c(
  "readr",
  "readxl", # Read file
  #"e1071",
  #"lubridate",
  #"scales",
  "dplyr",
  #"tidyr",
  "tidyverse", # Data manipulation
  #"Hmisc",
  #"DescTools",
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
data <- data.frame(
  Candidate = c("A", "B", "C", "D", "E", "F"),
  Years_Graduated = c(5, 3, 8, 6, 2, 7),
  Years_Last_Job = c(7, 4, 6, 5, 3, 9),
  Years_Studied = c(6, 5, 7, 8, 2, 4)
)

## Add Candidates as index
rownames(data) <- data[,1]
data <- data[,-1]

## Calculate Distance Matrix
d <- dist(data, method = "euclidean")
d

## Define each clusters for each method
HC_single <- hclust(d, method = "single" )
HC_complete <- hclust(d, method = "complete" )
HC_singleAverage <- hclust(d, method = "average" )
HC_centroid <- hclust(d, method = "centroid" )
HC_ward <- hclust(d, method = "ward.D" )

## Plot Dendograms
plot(HC_single, cex = 0.6, hang = -1)
plot(HC_complete, cex = 0.6, hang = -1)
plot(HC_singleAverage, cex = 0.6, hang = -1)
plot(HC_centroid, cex = 0.6, hang = -1)
plot(HC_ward, cex = 0.9, hang = -1)

## Choosing groups
HC_chosen <- HC_ward
plot(HC_chosen, cex = 0.6, hang = -1)
rect.hclust(HC_chosen, k = 2)

### Mark groups
groups <- cutree(HC_chosen, k = 2)


## Elbow
elbow <- fviz_nbclust(data, 
                      FUN = hcut, 
                      method = "wss", 
                      hc_method = "ward.D",
                      k.max = 5)
elbow 

wss_values <- elbow$data
wss_values

## Sillhouete
silhouette_values <- silhouette(groups, d)
rownames(silhouette_values) <- rownames(data)
#value of each data point for the number of groups chosen
silhouette_values
#chart
fviz_silhouette(silhouette_values, label = TRUE)

fviz_nbclust(data, 
             FUN = hcut, 
             method = "silhouette", 
             hc_method = "ward.D",
             k.max = 5)


# values of each point of sillhouete chart - library(NbClust) 
silhouette_sugestion <- fviz_nbclust(data, 
                                    FUN = hcut, 
                                    method = "silhouette", 
                                    hc_method = "ward.D",
                                    k.max = 5)
silhouette_sugestion$data

