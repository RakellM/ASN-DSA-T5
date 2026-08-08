library(sf)
library(dplyr)
library(ggplot2)
library(tmap)
library(leaflet)
library(classInt)
library(RColorBrewer)

# Desativar o uso de geometria esférica (S2) para evitar erros de intersecção

sf::sf_use_s2(FALSE)

# Carregar dados das Localidades (Distritos)

setwd("")
localidade <- read_sf("distritos_sp.gpkg")

# Carregar dados dos Endereços (Pontos)

setwd("")
enderecos <- read_sf("pontos_sp.shp")

st_crs(localidade)
st_crs(enderecos)

# Visualizações rápidas iniciais

plot(st_geometry(localidade))
plot(st_geometry(enderecos))

# Filtrar apenas endereços domiciliares (COD_ESP == 1)

domicilios <- enderecos %>%
  dplyr::filter(COD_ESP == 1)

st_crs(domicilios)

# Visualizações rápidas dos domicílios

ggplot() + 
  geom_sf(data = domicilios)

tm_shape(domicilios) +
  tm_dots(size = 0.1, fill = "red")

# Conta quantos domicílios estão em cada distrito (usando EPSG 3857)

ggplot() + 
  geom_sf(data = localidade) +
  geom_sf(data = domicilios)
  
inter <- st_intersects(st_transform(localidade, 3857), domicilios)
localidade$qtd_domicilios <- lengths(inter)

sum(localidade$qtd_domicilios)
nrow(domicilios)

# Identificar domicílios que ficaram fora de qualquer distrito

domicilios$local <- st_within(domicilios, st_transform(localidade, 3857))
valida <- domicilios %>%
  dplyr::filter(lengths(local) == 0)

nrow(valida) # Quantidade de domicílios não associados

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(data = st_transform(localidade,4326), opacity = 0.6,  weight = 1.5) %>% 
  addCircles(data = st_transform(valida,4326), weight = 5, color = 'red', opacity  = 1)


# Mapas temáticos

# Usando o comando nativo plot()

plot(localidade["qtd_domicilios"])

localidade %>%
  st_drop_geometry() %>%
  summarise(
    Min = min(qtd_domicilios),
    Media = mean(qtd_domicilios),
    Max = max(qtd_domicilios)
  )

# Testando diferentes quebras (breaks) no plot

plot(localidade["qtd_domicilios"], breaks = "sd", nbreaks = 7, pal = hcl.colors(7, "Blues"))
plot(localidade["qtd_domicilios"], breaks = "equal", nbreaks = 7, pal = hcl.colors(7, "Blues"))
plot(localidade["qtd_domicilios"], breaks = "quantile", nbreaks = 7, pal = hcl.colors(7, "Blues"))
plot(localidade["qtd_domicilios"], breaks = "kmeans", nbreaks = 7, pal = hcl.colors(7, "Blues"))
plot(localidade["qtd_domicilios"], breaks = "jenks", nbreaks = 7, pal = hcl.colors(7, "Blues"))

# Plot customizado com legenda e paleta de cores

plot(
  localidade["qtd_domicilios"], 
  key.pos = 1, 
  breaks = "kmeans", 
  nbreaks = 7,
  pal = rev(brewer.pal(7, "RdYlBu")),
  graticule = TRUE, 
  axes = TRUE,
  main = "Quantidade de domicilios por area"
)

# Usando ggplot2

# Escala contínua básica

ggplot() + 
  geom_sf(data = localidade, aes(fill = qtd_domicilios))

# Escala contínua customizada

ggplot(localidade, aes(fill = qtd_domicilios)) +
  geom_sf() +
  scale_fill_distiller("Qtd", palette = "RdYlBu") +
  theme_minimal() +
  labs(
    title = "Quantidade de domicílios por área",
    subtitle = "Município de São Paulo",
    caption = "Coordenadas Geográficas/IBGE"
  ) 

# Escala categórica (Quebras por Jenks)

classes <- classIntervals(localidade$qtd_domicilios, n = 7, style = "jenks")$brks
localidade$qtd_domicilios_classe <- cut(localidade$qtd_domicilios, classes, include.lowest = TRUE)

ggplot(localidade, aes(fill = qtd_domicilios_classe)) +
  geom_sf() +
  scale_fill_brewer("Qtd", palette = "RdYlBu", direction = -1) +
  theme_minimal() +
  labs(title = "Quantidade de domicílios por área",
       subtitle = "Município de São Paulo",
       caption = "Coordenadas Geográficas/IBGE") 

# Usando tmap (Thematic Maps)

# Estilo Quantile

localidade %>%
  tm_shape() +
  tm_fill(col = "qtd_domicilios", title = "# Domicílios", style = "quantile", n = 5) +
  tm_borders(lwd = 0.5) +
  tm_style("gray") +
  tm_layout(main.title = "Quantidade de domicílios por área",
            main.title.position = "center",
            main.title.size = 1,
            legend.position = c("right", "bottom")) 

# Estilo Jenks

localidade %>%
  tm_shape() +
  tm_fill(col = "qtd_domicilios", title = "# Domicílios", style = "jenks", n = 7) +
  tm_borders(lwd = 0.5) +
  tm_style("gray") +
  tm_layout(main.title = "Quantidade de domicílios por área",
            main.title.position = "center",
            main.title.size = 1,
            legend.position = c("right", "bottom")) 

# Estilo Equal

localidade %>%
  tm_shape() +
  tm_fill(col = "qtd_domicilios", title = "# Domicílios", style = "equal", n = 5) +
  tm_borders(lwd = 0.5) +
  tm_style("gray") +
  tm_layout(main.title = "Quantidade de domicílios por área",
            main.title.position = "center",
            main.title.size = 1,
            legend.position = c("right", "bottom")) 


# Mapas interativos
# O Leaflet exige coordenadas geográficas (WGS84 / EPSG:4326)

# Transformando os dados para a projeção do Leaflet

domicilios <- st_transform(domicilios, 4326)
localidade <- st_transform(localidade, 4326)

# Mapa básico apenas com os pontos de domicílios

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addCircles(data = domicilios, weight = 1)

# Mapa combinando distritos e os domicílios validados (com popup)

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(data = localidade, opacity = 0.6, weight = 1.5) %>% 
  addCircles(data = domicilios, weight = 5, color = 'red', opacity = 1, popup = ~as.character(COD_ESP))

# Filtros específicos por distrito

localidade %>%
  dplyr::filter(NM_DIST == 'Perdizes') %>% 
  leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(weight = 10, popup = ~NM_DIST)

localidade %>%
  dplyr::filter(NM_DIST == 'Morumbi') %>% 
  leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(weight = 10, popup = ~NM_DIST)

# Leaflet Coroplético (Temático por área)

# Configuração da paleta de cores

pal <- colorNumeric(palette = "viridis", domain = localidade$qtd_domicilios)

## pal <- colorBin(palette = "YlOrRd", domain = localidade$qtd_domicilios, bins = c(0, 200, 500, 1000, Inf))

## pal <- colorQuantile(palette = "viridis", domain = localidade$qtd_domicilios, n = 4, probs = seq(0, 1, length.out = 5))

# Mapa final interativo com legenda

localidade %>%
  leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(weight = 1,
              color = "white",
              fillOpacity = 0.7,
              fillColor = ~pal(qtd_domicilios)) %>% 
  addLegend("topright",
            pal = pal,
            values = ~qtd_domicilios,
            title = "Quantidade de domicílios<br>por área",
            opacity = 1)
