library(sf)
library(dplyr)

# O fluxo de construção de um objeto espacial no sf segue a lógica:
  # sfg (Geometria pura) -> sfc (Coleção com CRS) -> sf (Geometria + Atributos)


# Simple Feature Geometry (sfg) - Geometrias simples

# Criando pontos individuais (X = Longitude, Y = Latitude)

p1 <- st_point(c(-46.656139, -23.561571)) 
p2 <- st_point(c(-46.663983, -23.587389)) 
p3 <- st_point(c(-46.632970, -23.550835)) 

p1
class(p1)


# Simple Feature Collection (sfc) - Agrupamento de geometries + CRS

# Agrupando vários sfg em um sfc (ainda sem CRS)

pontos <- st_sfc(p1, p2, p3)
pontos
class(pontos)
plot(pontos, axes = TRUE)

# Atribuindo um Sistema de Referência de Coordenadas (WGS 84 - EPSG 4326)

pontos <- st_sfc(p1, p2, p3, crs = 4326)
pontos
plot(pontos, axes = TRUE)


# Objeto Simple Feature (sf) - Associação da Geometria com Tabela de Dados

# Criando a tabela de atributos (Dataframe)

Id <- c(1, 2, 3)
Nome <- c("Masp", "Ibirapuera", "Centro")
informacoes <- data.frame(Id, Nome)
class(informacoes)

# Unindo os atributos (informacoes) à geometria (pontos)

mapa <- st_sf(informacoes, geometry = pontos)
class(mapa) # Classe dupla: simple feature e dataframe
mapa

# Acessando colunas e geometrias isoladas

mapa$Nome
st_geometry(mapa)

# Diferença entre plotar o objeto completo vs apenas a geometria

plot(mapa, axes = TRUE)             # Cria mapas temáticos para cada coluna
plot(st_geometry(mapa), axes = TRUE) # Plota apenas o desenho dos pontos


# Salvando os arquivos

setwd("")

st_write(mapa, "mapa.shp", delete_layer = TRUE)
st_write(mapa, "mapa.kml", delete_layer = TRUE)
st_write(mapa, "mapa.gpkg", delete_layer = TRUE)


# Importando pontos (Endereços)

setwd("")
pontos_sp <- read_sf("pontos_sp.shp")

# Importando polígonos (Distritos)

setwd("")
distritos <- read_sf("distritos_sp.gpkg")

class(distritos) # Objeto espacial + dados (sf/data.frame)


# Visualização  

plot(pontos)
plot(st_geometry(pontos))
plot(st_geometry(distritos))


# Verificando o sistema de projeção

st_crs(pontos) 
st_crs(distritos) 


# Convertendo a projeção

plot(st_geometry(distritos), axes = TRUE) # Projeção baseada em graus

distritos <- st_transform(distritos, 3857)

plot(st_geometry(distritos), axes = TRUE) # Projeção baseada em metros

st_crs(distritos)

inter_clientes_distritos <- st_intersection(pontos,distritos)


# Usando %>% 

class(pontos_sp)

plot(st_geometry(pontos_sp))

predios_construcao <- pontos_sp %>%
  dplyr::filter(COD_ESP == 7) #Edificação em construção

plot(st_geometry(predios_construcao))
plot(st_geometry(distritos), add = TRUE)

st_crs(predios_construcao)
st_crs(distritos)

predios_construcao <- st_transform(predios_construcao, 4326)
distritos <- st_transform(distritos, 4326)

plot(st_geometry(predios_construcao))
plot(st_geometry(distritos), add = TRUE)