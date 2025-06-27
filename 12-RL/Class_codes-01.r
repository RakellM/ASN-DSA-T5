library("dplyr")

gasto <- c(4,5,6,12,13,14,8,9,10)
faixa_etaria <- c("jovem","jovem","jovem","adulto","adulto","adulto","idoso","idoso","idoso")

df <- data.frame(gasto, faixa_etaria)

df %>% 
  ggplot() +
  geom_point(aes(x = faixa_etaria, y = gasto))

df$Y_chapeu <- c(5,5,5,13,13,13,9,9,9)  

df %>% 
  ggplot() +
  geom_point(aes(x = gasto , y = gasto)) +
  geom_point(aes(x = gasto , y = Y_chapeu), color="blue")
