library(dplyr)
library(tidyr)
library(sf)
library(tmap)
library(cleangeo)   # Inspeção e limpeza de topologia espacial
library(spdep)      # Criação de matrizes de vizinhança e testes de Moran
library(spatialreg) # Modelos de Regressão Espacial
library(GWmodel)    # Regressão Local Ponderada

sf::sf_use_s2(FALSE)

setwd("")


# Carregamento dos dados

distritos <- read_sf("distritos_sp.gpkg") %>%  
  st_transform(3857)

pontos <- read_sf("pontos.gpkg") %>%  
  st_transform(3857) %>% 
  sample_n(20000)

resultado_sf <- distritos %>%
  st_join(pontos, join = st_intersects) %>%
  filter(!is.na(CLASSE)) %>%
  group_by(NM_DIST, CLASSE) %>%
  summarise(consumo_medio = mean(CONSUMO_MEDIO, na.rm = TRUE),
            .groups = "drop") %>%
  pivot_wider(names_from = CLASSE,
              values_from = consumo_medio,
              names_prefix = "Consumo_",
              values_fill = 0)


# Mapas Temáticos das Variáveis de Estudo

tmap_mode("plot")

resultado_sf %>%
  tm_shape() +
  tm_fill(col = "Consumo_Residencial1", title = "Consumo Médio", style = "jenks",
          n = 6, palette = "Blues", colorNA = NA, textNA = "") +
  tm_borders(lwd = 0.5) +
  tm_layout(main.title = "Consumo Médio", main.title.position = "center")

resultado_sf %>%
  tm_shape() +
  tm_fill(col = "Consumo_Residencial2", title = "Consumo Médio", style = "jenks",
          n = 6, palette = "Blues", colorNA = NA, textNA = "") +
  tm_borders(lwd = 0.5) +
  tm_layout(main.title = "Consumo Médio", main.title.position = "center")


# Autocorrelação Espacial Global (Índice de Moran)

### Matriz de vizinhança e pesos espaciais - Definir contiguidade espacial

vizinhanca <- poly2nb(resultado_sf)

### Visualizar a rede de conexões dos vizinhos

plot(st_geometry(resultado_sf), border = "gray")
plot(vizinhanca, coords = st_coordinates(st_centroid(st_geometry(resultado_sf))), 
     cex = 0.6, col = "red", add = TRUE)

### Criar matrizes de pesos espaciais (W)

vizinhanca_matriz <- nb2listw(vizinhanca, style = "W", zero.policy = TRUE) 

###  Criar a variável defasada espacialmente (Spatial Lag)

resultado_sf$LAG <- lag.listw(vizinhanca_matriz, var = resultado_sf$Consumo_Residencial1)

### Gráfico de Dispersão de Moran

plot(resultado_sf$LAG, resultado_sf$Consumo_Residencial1, 
     xlab = "Consumo", ylab = "Lag Consumo")

lm(resultado_sf$LAG ~ resultado_sf$Consumo_Residencial1)

### Testes Formais do Índice de Moran Global

moran.test(resultado_sf$Consumo_Residencial1, listw = vizinhanca_matriz)
moran.test(resultado_sf$Consumo_Residencial2, listw = vizinhanca_matriz)
moran.test(resultado_sf$Consumo_Comercial, listw = vizinhanca_matriz)

### Correlograma Espacial (Avaliar autocorrelação em múltiplas ordens de vizinhos)

correlograma <- sp.correlogram(neighbours = vizinhanca, style = "W",
                               var = resultado_sf$Consumo_Residencial1, 
                               order = 5, method = "I")
plot(correlograma)


# Indice Local de Moran

### Contribuição individual para o cálculo global do Indice Global de Moran

plot(scale(resultado_sf$Consumo_Residencial1), scale(resultado_sf$LAG),
     xlab = "Z-Densidade", ylab = "Z-Lag Densidade")
abline(h = 0, v = 0, lty = 2, col = "red")

### Autocorrelação Espacial Local (LISA)

moran_local <- localmoran(x = resultado_sf$Consumo_Residencial1, listw = vizinhanca_matriz)
moran_local_df <- as.data.frame(moran_local)

resultado_sf$moran   <- moran_local_df$Ii
resultado_sf$moran_p <- moran_local_df$`Pr(z != E(Ii))`

### Montando as categorias dos quadrantes para o Mapa LISA

quadrante1 <- factor(resultado_sf$Consumo_Residencial1 < mean(resultado_sf$Consumo_Residencial1), labels=c("Alto", "Baixo")) 
quadrante2 <- factor(resultado_sf$LAG < mean(resultado_sf$LAG), labels=c("Alto", "Baixo")) 

resultado_sf$quadrante <- paste(quadrante1, quadrante2)

### Mapa LISA

tm_shape(resultado_sf) +
  tm_fill("quadrante", palette = c("red", "lightblue", "blue", "pink", "white"),
                                   colorNA = NA, textNA = "", 
          title = "Clusters LISA") +
  tm_borders(lwd = 0.3)

### Mapa LISA apenas com os setores estatisticamente significativos (p <= 0.05)

resultado_sf <- resultado_sf %>%
  dplyr::mutate(quadrante_lisa = ifelse(moran_p <= 0.05, as.character(quadrante), "Não Significativo"))

tm_shape(resultado_sf) +
  tm_fill("quadrante_lisa", palette = c("red", "lightblue", "blue", "pink", "white"),
          colorNA = NA, textNA = "", 
          title = "Clusters LISA") +
  tm_borders(lwd = 0.3)


# Modelos de Regressão

### Regressão Linear Clássica

regressao <- lm(data = resultado_sf, Consumo_Residencial1 ~ Consumo_Residencial2)
summary(regressao)

### Análise dos resíduos

resultado_sf$residuos <- residuals(regressao)

tm_shape(resultado_sf) +
  tm_fill("residuos", style = "quantile", palette = "-RdBu",
          colorNA = NA, textNA = "", 
          title = "Resíduos")

### Teste de Moran nos resíduos (Se p < 0.05, podemos usar modelos de regressão espacial)

lm.morantest(regressao, listw = vizinhanca_matriz)

### Modelo Spatial Lag

regressao_espacial_lag <- lagsarlm(data = resultado_sf, 
                                   Consumo_Residencial1 ~ Consumo_Residencial2, listw = vizinhanca_matriz) 
summary(regressao_espacial_lag)
summary(regressao_espacial_lag, Nagelkerke = TRUE) # Cálculo do Pseudo-R²