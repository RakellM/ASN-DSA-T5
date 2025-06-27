#items in work space
ls()

# remove all
rm(list=ls())

## Libraries
# Packages
load.pks <- c(
  "readr",
  "plotly",
  #"e1071",
  #"lubridate",
  #"scales",
  "dplyr",
  #"tidyr",
  #"Hmisc",
  #"DescTools",
  #"esquisse",
  #"gridExtra",
  "kableExtra",
  #"ClusterR",
  #"cluster",
  #"pivottabler",
  #"ggcorrplot",
  #"corrplot",
  #"MASS",
  #"RMySQL",
  #"DBI",
  #"htmlwidgets",
  "knitr",
  #"bigrquery",
  "PerformanceAnalytics" ,
  "GGally",    # Correlation charts
  "patchwork", # Combine plots
  "ggplot2"
)

#-- Install
# lapply(load.pks, install.packages, character.only = TRUE)

#-- Load
#lapply(load.pks, require, character.only = TRUE)

#-- Install
is_installed <- load.pks %in% rownames(installed.packages())
if(any(is_installed == FALSE)){
  install.packages(load.pks[!is_installed])
}
#-- Load packages
sapply(load.pks, require, character.only = TRUE)

# Import Data
bodyfat_raw <- read.csv("C:/Users/kel_m/OneDrive/Studies/2024_ASN-DSAT5/12_RL/Reg_Linear/dados/Bodyfat.csv")

head(bodyfat_raw)

str(bodyfat_raw)

# EDA - Exploratory Data Analysis
## Descriptive Statistics
summary(bodyfat_raw)

## Type of variables
sapply(bodyfat_raw, class)

## Check for NA
colSums(is.na(bodyfat_raw))

## Removing Y < 3
dfBodyfat <- bodyfat_raw %>%
  filter(bodyfat >= 3)

## Check Distribution (Y)
### Histogram
g1 <- ggplot(dfBodyfat, aes(x = bodyfat)) +
  geom_histogram(fill = "#2c3e50", color = "white") +
  labs(
    title = "Histogram",
    x = "Bodyfat",
    y = "Frequency"
  ) +
  theme_minimal()

### Box-Plot
g2 <- ggplot(dfBodyfat, aes(y = bodyfat)) +
  geom_boxplot(fill = "#3498db", color = "#2c3e50", outlier.color = "red", outlier.shape = 16) +
  labs(
    title = "BoxPlot",
    y = "Bodyfat"
  ) +
  theme_minimal()

g1 + g2


## Check Distribution (Xs)
plot_hist_box <- function(data, var_name) {
  p_hist <- ggplot(data, aes_string(x = var_name)) +
    geom_histogram(aes(y = ..density..), fill = "skyblue", bins = 30, color = "black") +
    geom_density(color = "red", size = 1) +
    theme_minimal() +
    ggtitle(paste("Histogram & Density -", var_name))
  
  p_box <- ggplot(data, aes_string(y = var_name)) +
    geom_boxplot(fill = "lightgreen") +
    coord_flip() +
    theme_minimal() +
    ggtitle(paste("Boxplot -", var_name))
  
  # Combine vertically
  p_combined <- p_hist / p_box
  return(p_combined)
}

### Filter numeric variables
numeric_vars <- names(dfBodyfat)[sapply(dfBodyfat, is.numeric)]

### Plot all
for (var in numeric_vars) {
  print(plot_hist_box(dfBodyfat, var))
}

## Correlation between numeric variables
ggpairs(dfBodyfat, 
        lower = list(continuous = wrap("smooth", alpha = 0.3, size = 0.5)),
        upper = list(continuous = wrap("cor", size = 2)),
        title = "Correlation between numeric variables")


# removendo Y para s
dfBodyfatXs <- dfBodyfat %>%
  select(-bodyfat) %>%
  filter(Height >50)

## Correlation between numeric variables
ggpairs(dfBodyfatXs, 
        lower = list(continuous = wrap("smooth", alpha = 0.3, size = 0.5)),
        upper = list(continuous = wrap("cor", size = 2)),
        title = "Correlation between numeric variables")


chart.Correlation(dfBodyfatXs, histogram = TRUE)

### Correlation matrix
cor_matrix <- cor(dfBodyfat, use = "complete.obs", method = "pearson")
print(round(cor_matrix, 3))

# MODEL FULL
model_full <- lm(bodyfat ~ ., data = dfBodyfat )
summary(model_full)

# MODEL (After Multicolineary Analysis)
## Removing variables after multicolinearity analisis
dfBodyfat_model <- dfBodyfat %>%
  select(-Hip, -Chest, -Weight, -Density, -Neck, -Thigh, -Knee)

### Correlation matrix
cor_matrix <- cor(dfBodyfat_model, use = "complete.obs", method = "pearson")
print(round(cor_matrix, 3))
chart.Correlation(dfBodyfat_model, histogram = TRUE)

## Model full
model1_full <- lm(bodyfat ~ ., data = dfBodyfat_model )
summary(model1_full)

## Modelo - Biceps
model1_A <- lm(bodyfat ~ . -Biceps, data = dfBodyfat_model )
summary(model1_A)

## Modelo - Biceps - Ankle
model1_B <- lm(bodyfat ~ . -Biceps -Ankle, data = dfBodyfat_model )
summary(model1_B)

## Model Stepwise
model1_step <- step(model1_full, direction="both")
summary(model1_step)

# ERROR ANALYSIS
dfBodyfat_model_R <- dfBodyfat_model
dfBodyfat_model_R$residue <- resid(model1_B)
dfBodyfat_model_R$fitted <- fitted(model1_B)


## Chart 1: Residue vs Fitted
p1 <- ggplot(dfBodyfat_model_R, aes(x = fitted, y = residue)) +
  geom_point(alpha = 0.6, color = "#2980b9") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Resíduos vs Valores Ajustados",
       x = "Valores Ajustados",
       y = "Resíduos") +
  theme_minimal()

## Chart 2: Histograma
p2 <- ggplot(dfBodyfat_model_R, aes(x = residue)) +
  geom_histogram(fill = "#27ae60", color = "white") +
  labs(title = "Histograma dos Resíduos",
       x = "Resíduos",
       y = "Frequência") +
  theme_minimal()

## Chart 3: QQ-Plot
p3 <- ggplot(dfBodyfat_model_R, aes(sample = residue)) +
  stat_qq(color = "#8e44ad") +
  stat_qq_line(color = "red", linetype = "dashed") +
  labs(title = "QQ-Plot dos Resíduos",
       x = "Quantis Teóricos",
       y = "Quantis Amostrais") +
  theme_minimal()


p1 / p2 / p3 + plot_layout(heights = c(1, 1, 1))

print(plot_hist_box(dfBodyfat_model_R, dfBodyfat_model_R$residue))


## Normality test
ks.test(dfBodyfat_model_R$residue, "pnorm", 
        mean = mean(dfBodyfat_model_R$residue), 
        sd = sd(dfBodyfat_model_R$residue))


