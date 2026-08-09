library(sf)
library(dplyr)
library(leaflet)

# Desativar geometria esférica para evitar erros em operações de intersecção

sf::sf_use_s2(FALSE)

# Carregar Localidades (Polígonos)

setwd("")
localidade <- read_sf("distritos_sp.gpkg")
st_crs(localidade)
plot(st_geometry(localidade))

# Carregar Endereços (Pontos)

setwd("")
enderecos <- read_sf("pontos_sp.shp")
st_crs(enderecos)
plot(st_geometry(enderecos))

# Filtragem de Subconjuntos de Pontos por Tipo (COD_ESP)

domicilios    <- enderecos %>% dplyr::filter(COD_ESP == 1) # Domicílios
estab_ensino  <- enderecos %>% dplyr::filter(COD_ESP == 4) # Ensino
estab_saude   <- enderecos %>% dplyr::filter(COD_ESP == 5) # Saúde

localidade <- st_transform(localidade, 3857)
domicilios <- st_transform(domicilios, 3857)

plot(st_geometry(localidade))
plot(st_geometry(domicilios), add = TRUE, col = "blue")


# Predicados Espaciais vs Overlay

# Transformando para SIRGAS 2000 antes das análises

estab_saude_utm <- st_transform(estab_saude, 31983)
localidade_utm  <- st_transform(localidade, 31983)
estab_ensino_utm <- st_transform(estab_ensino, 31983)

# Predicados Espaciais (Retornam relações lógicas / índices)

predicado <- st_within(estab_saude_utm, localidade_utm)
class(predicado)

st_intersects(estab_ensino_utm, localidade_utm)

st_contains(estab_ensino_utm, localidade_utm)
st_contains(localidade_utm, estab_ensino_utm)

st_touches(localidade_utm, localidade_utm)

st_equals(localidade_utm, localidade_utm)

# Overlay (Retornam novas geometrias modificadas)

overlay <- st_intersection(estab_saude_utm, localidade_utm)
class(overlay)

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addCircles(data = st_transform(overlay, 4326), weight = 10, color = 'red', opacity = 1, popup = ~NM_DIST)

diferenca <- st_difference(estab_saude_utm, localidade_utm)

uniao <- st_union(estab_saude_utm, localidade_utm)

# Cálculo de Área (Polígonos)

localidade <- st_transform(localidade, 3857)

localidade$area <- as.numeric(st_area(localidade) / 1000000) 

# Forma correta utilizando Sistema Projetado (SIRGAS 2000)

localidade <- st_transform(localidade, 31983)

localidade$area_km2 <- as.numeric(st_area(localidade)) / 1000000

# Centróides

localidade %>% 
  st_transform(31983) %>% 
  dplyr::select(NM_DIST) %>% 
  st_centroid() %>% 
  head(1)

localidade %>% 
  st_transform(4326) %>% 
  dplyr::select(NM_DIST) %>% 
  st_centroid() %>% 
  head(1)

localidade %>% 
  st_transform(4326) %>% 
  dplyr::select(NM_DIST) %>% 
  st_point_on_surface() %>% # Garante que o ponto fique dentro do polígono
  head(1)

# Área de Influência (BUFFER)

# Para criar o Buffer em metros, precisamos de projeção métrica (ex: 31983)

estab_ensino_utm <- st_transform(estab_ensino, 31983)
st_crs(estab_ensino)

plot(st_geometry(estab_ensino), axes = TRUE)  

# Criando raio de influência de 1000 metros (1 km)

area_influencia_utm <- st_buffer(estab_ensino, 1000)

plot(st_geometry(area_influencia_utm))
plot(st_geometry(estab_ensino), add = TRUE, col = "red")

# Visualizações no Leaflet (Exige EPSG 4326)

area_influencia_4326 <- st_transform(area_influencia_utm, 4326)
estab_ensino_4326 <- st_transform(estab_ensino_utm, 4326)

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(data = area_influencia_4326, opacity = 0.6, weight = 1.5) %>% 
  addCircles(data = estab_ensino_4326, weight = 10, color = 'red', opacity = 1)

area_influencia_4326 %>%
  dplyr::filter(ID == 2) %>%
  leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(color = 'red', weight = 2) %>% 
  addPolygons(data = st_transform(localidade, 4326), fillOpacity = 0.2, weight = 1)

# Intersecção: Domicílios dentro da Área de Influência

# Convertendo ambos para a mesma projeção métrica para realizar o cruzamento

inter_domicilios_areainfl <- st_intersection(
  st_transform(domicilios, 3857),
  st_transform(area_influencia_utm, 3857))

# Analisando pontos duplicados (Clientes que estão em mais de uma área de influência)

inter_domicilios_areainfl %>% 
  sf::st_drop_geometry() %>%
  dplyr::group_by(ID) %>% 
  dplyr::summarise(n = n()) %>% 
  dplyr::arrange(desc(n)) %>% 
  head(5)

# Intersecção: Polígonos x Polígonos (Área de influência x Distritos)

inter_areainfl_localidade <- st_intersection(
  st_transform(area_influencia_utm, 4326),
  st_transform(localidade, 4326)) %>%
  dplyr::select(ID, NM_DIST) %>% 
  dplyr::arrange(ID)

inter_areainfl_localidade %>%
  leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(color = 'red', weight = 2)

# Cálculo de distâncias

st_distance(
  st_transform(estab_ensino, 4326),
  st_transform(estab_saude, 4326))

st_distance(
  st_transform(estab_ensino, 31983),
  st_transform(estab_saude, 31983))

st_distance(
  st_transform(estab_ensino, 3857),
  st_transform(estab_saude, 3857))

# Estruturando as matrizes de Distância

# Matriz via Projeção Geográfica

dist_pg <- as.data.frame(
  st_distance(st_transform(estab_ensino, 4326), st_transform(estab_saude, 4326)))

colnames(dist_pg) <- estab_saude[['ID']]
rownames(dist_pg) <- estab_ensino[['ID']] 

# Matriz via Projeção Plana

dist_pp <- as.data.frame(
  st_distance(st_transform(estab_ensino, 3857), st_transform(estab_saude, 3857)))  

colnames(dist_pp) <- estab_saude[['ID']]
rownames(dist_pp) <- estab_ensino[['ID']]