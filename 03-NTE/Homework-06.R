
options(scipen = 20)

#items in work space
ls()

# remove all
rm(list=ls())

## Libraries
# Packages
load.pks <- c(
  "readr",
  "ggplot2",
  "plotly",
  "e1071",
  "lubridate",
  "scales",
  "dplyr",
  "tidyr",
  "Hmisc",
  "DescTools",
  "esquisse",
  "gridExtra",
  "kableExtra",
  "ClusterR",
  "cluster",
  "pivottabler",
  "ggcorrplot",
  "corrplot",
  #"MASS",
  #"RMySQL",
  #"DBI",
  "htmlwidgets",
  "shiny",
  "knitr",
  "bigrquery"
)

#-- Load
lapply(load.pks, require, character.only = TRUE)

#remotes::install_github("juba/rmdformats")
#remotes::install_github("glin/reactable")
#remotes::install_github("juba/bookup-html")

data <- read.csv(".\\ASN-DSA-T5\\03-NTE\\data\\notas_comb.csv")


ggplot(data, aes(x = pais, y = x, fill = pais)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot de Horas de Estudo por País", x = "País", y = "Horas de Estudo")

ggplot(data, aes(x = pais, y = y, fill = pais)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot de Notas nos testes por País", x = "País", y = "Notas nos Testes")

ggplot(data, aes(x = x, fill = pais)) +
  geom_histogram(binwidth = 10, color = "black") +
  facet_wrap(~ pais) +
  labs(
    title = "Histograma de Horas de Estudo por País",
    x = "Horas de Estudo",
    y = "Frequência"
  ) +
  theme_minimal()

ggplot(data, aes(x = y, fill = pais)) +
  geom_histogram(binwidth = 10, color = "black") +
  facet_wrap(~ pais) +
  labs(
    title = "Histograma de Notas nos testes por País",
    x = "Notas nos Testes",
    y = "Frequência"
  ) +
  theme_minimal()

library(ggplot2)

ggplot(data, aes(x = x, y = y, color = pais)) +
  geom_point() +
  facet_wrap(~ pais) +
  labs(
    title = "Gráfico de Dispersão x vs y por País",
    x = "Horas de Estudo",
    y = "Notas nos Testes"
  ) +
  theme_minimal()
