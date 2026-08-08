library(sf)
library(leaflet)
library(ggplot2)
library(dplyr)
library(sfcentral)   # Análise de centros geográficos e dispersão
library(spatstat)    # Necessário para a análise de Kernel
library(raster)

sf::sf_use_s2(FALSE)

setwd("")


# Carregamento dos dados

distritos <- read_sf("distritos_sp.gpkg") %>%  
  st_transform(3857)

pontos <- read_sf("pontos.gpkg") %>%  
  st_transform(3857) %>% 
  sample_n(1000)


# Boxplot de Consumo Médio por Classe

pontos %>% 
  ggplot(aes(x = CLASSE, y = CONSUMO_MEDIO, fill = CLASSE)) +
  geom_boxplot() +
  labs(title = "Consumo Médio por Classe",
       x = "Classe",
       y = "Consumo Médio") +
  theme_minimal() +
  theme(legend.position = "none")


# Mapa dos pontos + distritos

ggplot() + 
  geom_sf(data = distritos, fill = "gray95", color = "gray80") +
  geom_sf(data = pontos, aes(color = CLASSE), size = 0.8) +
  scale_color_brewer(palette = "Set1") +
  theme_minimal()


# Estatística Espacial Descritiva

### Centro Médio

media_co <- st_central_point(pontos %>% dplyr::filter(CLASSE == 'Comercial'), method = "mean") %>% st_transform(4326)
media_r1 <- st_central_point(pontos %>% dplyr::filter(CLASSE == 'Residencial1'), method = "mean") %>% st_transform(4326)
media_r2 <- st_central_point(pontos %>% dplyr::filter(CLASSE == 'Residencial2'), method = "mean") %>% st_transform(4326)

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addCircleMarkers(data = media_co, color = "red", radius = 6, group = "Comercial") %>%
  addCircleMarkers(data = media_r1, color = "green", radius = 6, group = "Residencial 1") %>%
  addCircleMarkers(data = media_r2, color = "blue", radius = 6, group = "Residencial 2") %>% 
  addLegend(position = "bottomright",
            colors = c("red", "green", "blue"),
            labels = c("Comercial", "Residencial 1", "Residencial 2"),
            title = "Centro Médio")

### Centro Mediano

mediana_co <- st_central_point(pontos %>% dplyr::filter(CLASSE == 'Comercial'), method = "median") %>% st_transform(4326)
mediana_r1 <- st_central_point(pontos %>% dplyr::filter(CLASSE == 'Residencial1'), method = "median") %>% st_transform(4326)
mediana_r2 <- st_central_point(pontos %>% dplyr::filter(CLASSE == 'Residencial2'), method = "median") %>% st_transform(4326)

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addCircleMarkers(data = mediana_co, color = "red", radius = 6) %>%
  addCircleMarkers(data = mediana_r1, color = "green", radius = 6) %>%
  addCircleMarkers(data = mediana_r2, color = "blue", radius = 6) %>% 
  addLegend(position = "bottomright",
            colors = c("red", "green", "blue"),
            labels = c("Comercial", "Residencial 1", "Residencial 2"),
            title = "Centro Mediano")

### Distância Padrão (Sem Pesos)

sdd_co <- st_sd_distance(pontos %>% dplyr::filter(CLASSE == 'Comercial')) %>% st_transform(4326)
sdd_r1 <- st_sd_distance(pontos %>% dplyr::filter(CLASSE == 'Residencial1')) %>% st_transform(4326)
sdd_r2 <- st_sd_distance(pontos %>% dplyr::filter(CLASSE == 'Residencial2')) %>% st_transform(4326)

leaflet() %>% 
  addProviderTiles(providers$CartoDB.Positron) %>%
  addPolygons(data = sdd_co, color = "red", opacity = 0.8, fillOpacity = 0.25) %>%
  addPolygons(data = sdd_r1, color = "green", opacity = 0.8, fillOpacity = 0.25) %>%
  addPolygons(data = sdd_r2, color = "blue", opacity = 0.8, fillOpacity = 0.25) %>% 
  addLegend(position = "bottomright",
            colors = c("red", "green", "blue"),
            labels = c("Comercial", "Residencial 1", "Residencial 2"),
            title = "Distância Padrão")

### Distância Padrão Ponderada (Pelo Consumo Médio)

pts_co <- pontos %>% dplyr::filter(CLASSE == 'Comercial')
pts_r1 <- pontos %>% dplyr::filter(CLASSE == 'Residencial1')
pts_r2 <- pontos %>% dplyr::filter(CLASSE == 'Residencial2')

sdd_co_w <- st_sd_distance(pts_co, weights = pts_co$CONSUMO_MEDIO) %>% st_transform(4326)
sdd_r1_w <- st_sd_distance(pts_r1, weights = pts_r1$CONSUMO_MEDIO) %>% st_transform(4326)
sdd_r2_w <- st_sd_distance(pts_r2, weights = pts_r2$CONSUMO_MEDIO) %>% st_transform(4326)

leaflet() %>% 
  addProviderTiles(providers$CartoDB.Positron) %>%
  addPolygons(data = sdd_co_w, color = "red", opacity = 0.8, fillOpacity = 0.25) %>%
  addPolygons(data = sdd_r1_w, color = "green", opacity = 0.8, fillOpacity = 0.25) %>%
  addPolygons(data = sdd_r2_w, color = "blue", opacity = 0.8, fillOpacity = 0.25) %>% 
  addLegend(position = "bottomright",
            colors = c("red", "green", "blue"),
            labels = c("Comercial", "Residencial 1", "Residencial 2"),
            title = "Distância Padrão Ponderada")

### Elipse de Desvio Padrão

sdd_co_e <- st_sd_ellipse(pontos %>% dplyr::filter(CLASSE == 'Comercial')) %>% st_transform(4326)
sdd_r1_e <- st_sd_ellipse(pontos %>% dplyr::filter(CLASSE == 'Residencial1')) %>% st_transform(4326)
sdd_r2_e <- st_sd_ellipse(pontos %>% dplyr::filter(CLASSE == 'Residencial2')) %>% st_transform(4326)

leaflet() %>% 
  addProviderTiles(providers$CartoDB.Positron) %>%
  addPolygons(data = sdd_co_e, color = "red", opacity = 0.8, fillOpacity = 0.25) %>%
  addPolygons(data = sdd_r1_e, color = "green", opacity = 0.8, fillOpacity = 0.25) %>%
  addPolygons(data = sdd_r2_e, color = "blue", opacity = 0.8, fillOpacity = 0.25) %>% 
  addLegend(position = "bottomright",
            colors = c("red", "green", "blue"),
            labels = c("Comercial", "Residencial 1", "Residencial 2"),
            title = "Elipse Desvio Padrão")


# Mapas de Densidade de Kernel

pts_co <- pontos %>% 
  dplyr::filter(CLASSE == 'Residencial1')

### Criar a "Janela" de estudo (Observation Window)

distritos_janela  <- as.owin(st_as_sf(distritos))

### Capturar coordenadas

pts_co_coords <- st_coordinates(pts_co)

### Construir o objeto de Processo Pontual (ppp)

pts_co_ppp <- ppp(
  x = pts_co_coords[, 1], 
  y = pts_co_coords[, 2], 
  window = distritos_janela,
  marks = pts_co$CONSUMO_MEDIO)

### Fixando um raio arbitrário de 1000 metros (1 km)

pts_co_kernel_1000 <- density(pts_co_ppp, sigma = 1000)
plot(pts_co_kernel_1000, main = "Kernel (Raio 1km) - Não Ponderado")

### Kernel ponderado (Consumo Médio)

pts_co_kernel_1000_w <- density(pts_co_ppp, sigma = 1000, weights = marks(pts_co_ppp))
plot(pts_co_kernel_1000_w, main = "Kernel (Raio 1km) - Ponderado por Consumo")

### Visualizando no leaflet

raster_1km <- raster(pts_co_kernel_1000)
raster_1km_w <- raster(pts_co_kernel_1000_w)

crs(raster_1km) <- CRS("+init=epsg:3857")
crs(raster_1km_w) <- CRS("+init=epsg:3857")

kernel_leaflet_1km <- projectRaster(raster_1km, crs = CRS("+init=epsg:4326"), method = "bilinear")
kernel_leaflet_1km_w <- projectRaster(raster_1km_w, crs = CRS("+init=epsg:4326"), method = "bilinear")

pal_1km <- colorNumeric(palette = "BuPu",domain = values(kernel_leaflet_1km),na.color = "transparent")
pal_1km_w <- colorNumeric(palette = "YlOrRd",domain = values(kernel_leaflet_1km_w),na.color = "transparent")

leaflet() %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  addRasterImage(kernel_leaflet_1km, colors = pal_1km, opacity = 0.6,group = "Kernel Simples (1km)") %>%
  addRasterImage(kernel_leaflet_1km_w, colors = pal_1km_w, opacity = 0.6, group = "Kernel Ponderado (Consumo)") %>%
  addLayersControl(overlayGroups = c("Kernel Simples (1km)", "Kernel Ponderado (Consumo)"))

### Cálculo do Raio Ótimo por Validação Cruzada

raio_otimo <- bw.ppl(pts_co_ppp)
print(raio_otimo)

### Testando diferentes funções matemáticas de suavização (Funções de Kernel) com raio ótimo

kernel_quartic  <- density(pts_co_ppp, sigma = raio_otimo, kernel = "quartic")
kernel_gaussian <- density(pts_co_ppp, sigma = raio_otimo, kernel = "gaussian")

plot(pts_co_kernel_1000, main = "Kernel (Raio 1km) - Não Ponderado")
plot(pts_co_kernel_1000_w, main = "Kernel (Raio 1km) - Ponderado por Consumo")
plot(kernel_quartic, main = "Kernel Quártico (Raio Ótimo)")
plot(kernel_gaussian, main = "Kernel Gaussiano (Raio Ótimo)")