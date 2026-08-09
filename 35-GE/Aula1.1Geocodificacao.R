library(dplyr)
library(sf)
library(googleway)
library(leaflet)
library(keyring)


# Lista de endereços

enderecos <- data.frame(
  enderecos = c(
    "Avenida Professor Campos de Oliveira, 146, São Paulo, SP, Brasil",
    "Rua Henri Dunant, 747, São Paulo, SP, Brasil",
    "Rua Horácio Bandieri, 33, São Paulo, SP, Brasil",
    "Rua Doutor Silvino Canuto Abreu, 153, São Paulo, SP, Brasil",
    "Rua Doutor Franklin Piza, 14, São Paulo, SP, Brasil",
    "Rua Leopoldo Couto de Magalhães Júnior, 807, São Paulo, SP, Brasil",
    "Rua Henrique Martins, 493, São Paulo, SP, Brasil",
    "Alameda Gabriel Monteiro da Silva, 398, São Paulo, SP, Brasil",
    "Rua Luís Murat, 400, São Paulo, SP, Brasil",
    "Rua Cotoxó, 1231, São Paulo, SP, Brasil",
    "Rua Félix Guilhem, 951, São Paulo, SP, Brasil",
    "Rua Narcisa Amália, 34, São Paulo, SP, Brasil",
    "Rua Jorge Rubens Neiva de Camargo, 423, São Paulo, SP, Brasil",
    "Rua das Tamareiras, 37, São Paulo, SP, Brasil",
    "Rua Simão da Matta, 479, São Paulo, SP, Brasil",
    "Avenida Jamaris, 1007, São Paulo, SP, Brasil",
    "Rua João Mafra, 44, São Paulo, SP, Brasil",
    "Rua das Giestas, 29, São Paulo, SP, Brasil",
    "Rua José Maronato, 190, São Paulo, SP, Brasil",
    "Rua Caconde, 128, São Paulo, SP, Brasil",
    "Rua Morgado Mateus, 154, São Paulo, SP, Brasil",
    "Avenida Angélica, 2100, São Paulo, SP, Brasil",
    "Rua Maestro Cardim, 42, São Paulo, SP, Brasil",
    "Rua Maris e Barros, 629, São Paulo, SP, Brasil",
    "Avenida do Estado, 6769, São Paulo, SP, Brasil",
    "Rua Frederico Alvarenga, 190, São Paulo, SP, Brasil",
    "Rua Xavantes, 719, São Paulo, SP, Brasil",
    "Avenida Marquês de São Vicente, 491, São Paulo, SP, Brasil",
    "Rua Alberto Savoy, 31, São Paulo, SP, Brasil",
    "Rua São Quirino, 900, São Paulo, SP, Brasil",
    "Rua Imbó, 428, São Paulo, SP, Brasil",
    "Rua Engenheiro Cestari, 631, São Paulo, SP, Brasil",
    "Rua Doutor Luiz Carlos, 1067, São Paulo, SP, Brasil",
    "Rua Aurora das Dores, 393, São Paulo, SP, Brasil",
    "Rua Crateús, 63, São Paulo, SP, Brasil",
    "SHCGN 705 Bloco A, 324, Brasília, DF, Brasil",
    "SHCGN 705 Bloco C, 324, Brasília, DF, Brasil",
    "CEP 05011-040, São Paulo, SP, Brasil",
    "Piquete, SP, Brasil",
    "CEP 12620-000, SP, Brasil",
    "Rua Primeiro de Maio, São Paulo, SP, Brasil",
    "Rua Primeiro de Maio, São Paulo, SP, Brasil"
  ),
  stringsAsFactors = FALSE
)


# Processo de geocodificação

# Realizar a consulta na API do Google para cada endereço

geocode <- lapply(
  enderecos$enderecos,
  function(x) {
    google_geocode(address = x, key = key_get("senha"))
  }
)

# Extração e estruturação das coordenadas lat/lon

coords <- lapply(seq_along(geocode), function(x) {
  
  # Caso o endereço não retorne resultados válidos
  if (is.null(geocode[[x]]$results) || length(geocode[[x]]$results) == 0) {
    return(
      data.frame(
        address = enderecos$enderecos[x],
        lat = NA,
        lon = NA,
        tipo = "SEM_RESULTADO",
        stringsAsFactors = FALSE
      )
    )
  }
  
  localizacao <- geocode[[x]]$results$geometry$location[1, ]
  tipo        <- geocode[[x]]$results$geometry$location_type[1]
  
  data.frame(
    address = enderecos$enderecos[x],
    lat = localizacao$lat,
    lon = localizacao$lng,
    tipo = tipo,
    stringsAsFactors = FALSE
  )
})

# Unificar a lista em um único DataFrame

df_coords <- bind_rows(coords)
df_coords
class(df_coords)

# Remover valores com NA para converter em objeto espacial (sf)

df_coords_validos <- df_coords %>% 
  dplyr::filter(!is.na(lat) & !is.na(lon))

# Converter para objeto espacial (WGS 84 / EPSG:4326)

pontos <- df_coords_validos %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326)

class(pontos)

# Mapa com os pontos geocodificados

pal <- colorFactor(palette = 'Set1',domain = df_coords$tipo)

leaflet(data = pontos) %>% 
  addTiles() %>% 
  addCircles( weight = 20, popup = ~as.character(address), color = ~pal(tipo)) %>%
  addLegend("bottomright", pal = pal, values = ~tipo)


# Salvando o arquivo

setwd("")

st_write(pontos,"pontos_geo.shp", delete_layer = TRUE)
st_write(pontos,"pontos_geo.kml", delete_layer = TRUE)
st_write(pontos,"pontos_geo.gpkg", delete_layer = TRUE)