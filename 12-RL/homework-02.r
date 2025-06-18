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
price_raw <- readRDS("C:/Users/kel_m/OneDrive/Studies/2024_ASN-DSAT5/12_RL/Homework/RL_Lista_02/precos.rds")

head(price_raw)

str(price_raw)

## Removing categorical for now
price_clean <- price_raw %>%
  select(-Heating_QC, -Season_Sold)

price <- as.data.frame(price_clean)

# EDA - Exploratory Data Analysis
## Descriptive Statistics
summary(price)

## Type of variables
sapply(price, class)

## Check for NA
colSums(is.na(price))

## Check Distribution (Sales Price)
### Histogram
ggplot(price, aes(x = SalePrice)) +
  geom_histogram(binwidth = 10000, fill = "#2c3e50", color = "white") +
  labs(
    title = "Sale Price Distribution",
    x = "Sale Price",
    y = "Frequency"
  ) +
  theme_minimal()

### Box-Plot
ggplot(price, aes(y = SalePrice)) +
  geom_boxplot(fill = "#3498db", color = "#2c3e50", outlier.color = "red", outlier.shape = 16) +
  labs(
    title = "Sale Price BoxPlot",
    y = "Sale Price"
  ) +
  theme_minimal()

## Correlation between numeric variables
ggpairs(price, 
        lower = list(continuous = wrap("smooth", alpha = 0.3, size = 0.5)),
        title = "Correlation between numeric variables")

## Dispersion between SalePrice and Gr_Liv_Area
ggplot(price, aes(x = Gr_Liv_Area, y = SalePrice)) +
  geom_point(color = '#2c3e50') +
  labs(
    title = "Above ground Living space vs Sale Price",
    x = "Above Ground Living (Square feet)",
    y = "Sale Price"
  ) +
  theme_minimal()

# Charts for Above Ground Living (Square feet)
### Histogram
ggplot(price, aes(x = Gr_Liv_Area)) +
  geom_histogram(binwidth = 30, fill = "#2c3e50", color = "white") +
  labs(
    title = "Above Ground Living (Square feet) Distribution",
    x = "Gr_Liv_Area",
    y = "Frequency"
  ) +
  theme_minimal()

### Box-Plot
ggplot(price, aes(y = Gr_Liv_Area)) +
  geom_boxplot(fill = "#3498db", color = "#2c3e50", outlier.color = "red", outlier.shape = 16) +
  labs(
    title = "Above Ground Living (Square feet) BoxPlot",
    y = "Gr_Liv_Area"
  ) +
  theme_minimal()

# Model
## 
model_full <- lm(SalePrice ~ ., data = price )
summary(model_full)

model_null <- lm(SalePrice ~ 1, data = price )
summary(model_null)

mBackward <- step(model_full, direction="backward")
summary(mBackward)

mForward <- step(model_null, scope=list(lower=model_null, upper=model_full), direction="forward")
summary(mForward)

mStepwise <- step(model_full, direction="both")
summary(mStepwise)

# Error Analisis
price_model <- price
price_model$residue <- resid(mStepwise)
price_model$fitted <- fitted(mStepwise)

## Chart 1: Residue vs Fitted
p1 <- ggplot(price_model, aes(x = fitted, y = residue)) +
  geom_point(alpha = 0.6, color = "#2980b9") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Resíduos vs Valores Ajustados",
       x = "Valores Ajustados",
       y = "Resíduos") +
  theme_minimal()

## Chart 2: Histograma
p2 <- ggplot(price_model, aes(x = residue)) +
  geom_histogram(binwidth = 10000, fill = "#27ae60", color = "white") +
  labs(title = "Histograma dos Resíduos",
       x = "Resíduos",
       y = "Frequência") +
  theme_minimal()

## Chart 3: QQ-Plot
p3 <- ggplot(price_model, aes(sample = residue)) +
  stat_qq(color = "#8e44ad") +
  stat_qq_line(color = "red", linetype = "dashed") +
  labs(title = "QQ-Plot dos Resíduos",
       x = "Quantis Teóricos",
       y = "Quantis Amostrais") +
  theme_minimal()


p1 / p2 / p3 + plot_layout(heights = c(1, 1, 1))

## Normality test
#shapiro.test(price_model$residue)
ks.test(price_model$residue, "pnorm", mean = mean(price_model$residue), sd = sd(price_model$residue))

