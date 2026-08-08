library(sf)
library(geobr)
library(ggplot2)
library(tmap)
library(dplyr)
library(classInt)
library(RColorBrewer)
library(leaflet)

sf::sf_use_s2(FALSE)

# Documentação do pacote disponível em: https://github.com/ipeaGIT/geobr

# Listar todas as tabelas disponíveis no geobr

tabelas <- list_geobr()

# Buscar códigos IBGE de municípios específicos

lookup_muni(name_muni = 'Campinas', year = 2022)
lookup_muni(name_muni = 'Piquete', year = 2022)
lookup_muni(name_muni = 'Aparecida', year = 2022) # Cidades homônimas em UFs diferentes

# Donwload dos dados

# Exemplo 1: Município de Campinas/SP (via código IBGE)

muni_campinas <- read_municipality(code_muni = 3509502, showProgress = FALSE, year = 2022)

ggplot() +
  geom_sf(data = muni_campinas) +
  theme_void()

# Exemplo 2: Todos os municípios do Estado de São Paulo (via sigla UF)

estado_sp <- read_municipality(code_muni = 'SP', showProgress = FALSE, year = 2022)

ggplot() +
  geom_sf(data = estado_sp) +
  theme_void()

# Exemplo 3: Região Metropolitana específica

rm_vale <- read_metro_area(code_state = 35, showProgress = FALSE, year = 2022) %>%
  dplyr::filter(type == 'RM do Vale do Paraíba e Litoral Norte (SP)')

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(data = rm_vale, opacity = 0.6, weight = 1.5)

# Definindo Campinas como a área de foco para análise das escolas

area_analise <- muni_campinas
municipios_nomes <- area_analise[["name_muni"]]

# Baixar o banco nacional de escolas e filtrar apenas para o município escolhido

escolas <- read_schools(showProgress = TRUE, year = 2025) %>%
  dplyr::filter(name_muni %in% municipios_nomes)

# Mapa básico de pontos de escolas sobre o município

ggplot(escolas) +
  geom_sf(data = area_analise, fill = "gray95") +
  geom_sf(size = 0.5, alpha = 0.6) +
  theme_minimal()

# Mapa diferenciando por Dependência Administrativa

table(escolas$tp_dependencia)

# 1 Federal
# 2 Estadual
# 3 Municipal
# 4 Privada

ggplot(escolas) +
  geom_sf(data = area_analise, fill = "gray95") +
  geom_sf(aes(color = tp_dependencia), size = 1) +
  theme_minimal()

ggplot(escolas) +
  geom_sf(data = area_analise, fill = "gray95") +
  geom_sf(aes(color = as.character(tp_dependencia))) +
  theme_minimal()

table(escolas$tp_situacao_funcionamento)

# 1 Ativa
# 2 Paralisada
# 3 Extinta

escolas %>%
  ggplot() +
  geom_sf(data = area_analise)+
  geom_sf(size = 1.5, aes(color = as.character(tp_dependencia))) +
  facet_wrap(~tp_situacao_funcionamento)


# Mapa Temático

area_grid = st_make_grid(st_transform(escolas, 3857),
                         c(1000,1000),
                         what = "polygons", square = TRUE)

area_grid_sf <- st_sf(area_grid) %>%
  mutate(grid_id = 1:length(lengths(area_grid)))

ggplot(escolas) +
  geom_sf(data = area_grid_sf) +
  geom_sf(size = 0.5)

area_grid_sf$n_escolas = lengths(st_intersects(area_grid_sf, st_transform(escolas, 3857)))

plot(area_grid_sf["n_escolas"])

pal <- colorNumeric(palette = "viridis",
                    domain = area_grid_sf$n_escolas)

st_transform(area_grid_sf, 4326)%>%
  leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(weight = 1,color = "white",fillOpacity = 0.7,fillColor = ~ pal(n_escolas)) %>% 
  addLegend("topright",
            pal = pal,
            values = ~n_escolas,
            title = "Quantidade de escolas<br>por área",
            opacity = 1)

pal <- colorBin(palette = "RdYlBu",
                domain = area_grid_sf$n_escolas,
                bins = c(1,3,5,7,9,100))

st_transform(area_grid_sf, 4326) %>%
  filter(n_escolas > 0) %>% 
  leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(weight = 1,color = "white",fillOpacity = 0.7,fillColor = ~ pal(n_escolas)) %>% 
  addLegend("topright",
            pal = pal,
            values = ~n_escolas,
            title = "Quantidade de escolas<br>por área",
            opacity = 1)


# Analise Setor Censitário

campinas_setor <- read_census_tract(code_tract = 3509502, year = 2022)

leaflet() %>% 
  addProviderTiles("OpenStreetMap.Mapnik") %>% 
  addPolygons(data = campinas_setor, opacity = 0.6, weight = 1.5)

st_join(escolas, campinas_setor["zone"]) %>%
  st_drop_geometry() %>%
  count(zone, name = "qtd_escolas")

st_join(escolas, campinas_setor["zone"]) %>% 
  filter(zone == 'Rural') %>% 
  leaflet() %>%
  addProviderTiles("CartoDB.Positron") %>%
  addCircleMarkers(
    radius = 4,
    color = "blue",
    stroke = FALSE,
    fillOpacity = 0.8)