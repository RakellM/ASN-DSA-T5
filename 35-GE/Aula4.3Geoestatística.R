library(dplyr)
library(sf)
library(tmap)
library(leaflet)
library(terra)
library(ggplot2)
library(gstat)      # Geoestatística (IDW e Krigagem)

sf::sf_use_s2(FALSE)

setwd("")


# Carregamento dos dados

pdv <- st_read("pdv_sp.gpkg") %>% 
  st_transform(3857)

distritos <- st_read("distritos_sp.gpkg") %>% 
  st_transform(3857)


# Visualização de quebras estatísticas do faturamento nos pontos

ggplot() + 
  geom_sf(data = distritos, fill = "gray95", color = "gray80") +
  geom_sf(data = pdv, aes(color = FATURAM), size = 2) +
  scale_color_viridis_c(option = "plasma", direction = -1) +
  theme_minimal()


# Interpolação por vizinhos mais próximos (Polígonos de Voronoi)

voronoi <- st_voronoi(st_union(pdv))

voronoi_sf <- st_collection_extract(voronoi, "POLYGON")

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(data = st_transform(voronoi_sf, 4326), opacity = 0.6, weight = 1.5, fillOpacity = 0.2) %>% 
  addCircles(data = st_transform(pdv, 4326), weight = 5, color = 'red', opacity = 1)


# Interpolação pelo inverso da distância (IDW)

grid <- st_make_grid(pdv,cellsize = 800, what = "centers")

grid <- st_sf(geometry = grid)

idw_result <- gstat::idw(formula = FATURAM ~ 1,
                         locations = pdv,
                         newdata = grid,
                         idp = 2)

plot(idw_result["var1.pred"])

### Otimização do peso

pesos <- seq(1, 5, by = 1)

rmse <- numeric(length(pesos))

for(i in seq_along(pesos)){
  cv <- krige.cv(FATURAM ~ 1,
                 locations = pdv,
                 nfold = nrow(pdv),
                 set = list(idp = pesos[i]))
  rmse[i] <- sqrt(mean(cv$residual^2))
  }

data.frame(idp = pesos, RMSE = rmse)

melhor_idp <- pesos[which.min(rmse)]
melhor_idp

idw_result <- gstat::idw(formula = FATURAM ~ 1,
                         locations = pdv,
                         newdata = grid,
                         idp = melhor_idp)

plot(idw_result["var1.pred"])

idw_raster <- rasterize(vect(idw_result),
                        rast(vect(idw_result),resolution = 800),
                        field = "var1.pred")

pal <- colorNumeric(palette = "viridis",
                    domain = values(idw_raster),
                    na.color = "transparent")

leaflet() %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  addRasterImage(idw_raster,colors = pal,opacity = 0.7) %>%
  addLegend(pal = pal,values = values(idw_raster),title = "IDW")

### Predição IDW para um ponto específico 

novo_ponto <- st_sf(
  geometry = st_sfc(st_point(c(350000, 7425000)),
                    crs = st_crs(pdv)))

pred <- idw(FATURAM ~ 1,
            locations = pdv,
            newdata = novo_ponto,
            idp = melhor_idp)

pred$var1.pred


# Krigagem

### Construção do Variograma Empírico

vgm <- variogram(FATURAM ~ 1, pdv)
plot(vgm, cex = 1.5, main = "Variograma Semivariância")

### Ajuste e seleção de modelos teóricos

vgm_esferico   <- fit.variogram(vgm, model = vgm("Sph"))
vgm_exponencial <- fit.variogram(vgm, model = vgm("Exp"))
vgm_gaussiano   <- fit.variogram(vgm, model = vgm("Gau"))

plot(vgm, vgm_esferico, main = "Ajuste Esférico")
plot(vgm, vgm_exponencial, main = "Ajuste Exponencial")
plot(vgm, vgm_gaussiano, main = "Ajuste Gaussiano")

### Ajuste automático testando múltiplos modelos simultâneos (Escolhe o de menor resíduo)

vgm_fit <- fit.variogram(vgm, vgm(c("Exp", "Gau", "Sph")))
plot(vgm, vgm_fit, main = "Melhor Modelo Escolhido")

### Execução da Krigagem na Grade

vgm_kriged <- krige(FATURAM ~ 1, pdv, grid, model = vgm_fit)

plot(vgm_kriged["var1.pred"], main = "Superfície Krigada (Predição)")

### Converter a saída da Krigagem para estrutura raster do pacote terra

raster_krig <- terra::rast(vgm_kriged["var1.pred"])

krig_vect <- vect(vgm_kriged)
r <- rast(ext(krig_vect), resolution = 800)
raster_krig <- rasterize(krig_vect,
                         r,
                         field = "var1.pred")

crs(raster_krig) <- "EPSG:3857"

plot(raster_krig, main = "Superfície Krigada (Predição)")

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>%
  addRasterImage(raster_krig, opacity = 0.7, colors = "Spectral")

### Predição por krigagem para o ponto isolado

predicao_kriged <- krige(formula = FATURAM ~ 1, locations = pdv, model = vgm_fit, newdata = novo_ponto)

predicao_kriged$var1.pred