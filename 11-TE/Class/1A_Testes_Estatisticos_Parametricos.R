# Testes estatisticos

library(readxl)
library(haven)
library(tidyverse)
library(car) #teste F de levene

# Legendas:
#  Explicacao Conteudo
## Dicas
### Pratica
#### Contexto Negocio

# Testes estatisticos sao muito uteis e necessarios para sanarmos respostas 
# rapidas e importantes para o entendimento do negocio.
# Tenha em mente que sempre que eu usar o termo "negocio" eu estou falando 
# sobre o contexto do problema que estivermos tratando. 
# Outros termos que sao importantes serem relembrados:
# - p-valor
# - nivel de confianca
# - nivel de significancia
# - intervalo de confianaca
# - teste bicaudal ou unilateral
# - Que o tamanho da rosquinha nao é o nerd que decide

# Testes estatisticos sao divididos em 2 grandes grupos:
# Testes parametricos
# envolvem parametros populacionais para dados quantitativos
# sao mais robustos (acendem a luz)
# tem regras para serem utilizados

# Testes nao parametricos
# as hipoteses sao formuladas sobre caracteristicas qualitativas da populacao
# sao menos robustos (acendem uma vela)
# sao testes livres de distribuicao (sem regras)

# Testes de Normalidade
# Conhecidos como teste de aderencia. Existem varios, conhecemos o:
# Teste de kolmogorov - Smirnov (KS) (usado quando n > 30)
# H0: a amostra eh normal
# H1: a amostra nao eh normal
# Teste de Shapiro Wilks (usado quando 4 <= n <= 2.000)
# H0: a amostra eh normal
# H1: a amostra nao eh normal

# Para apresenta-los geraremos um objeto com 1.000 linhas a partir de uma 
# distribuicao normal e aplicaremos os 2 testes para aprendermos os comandos
# na sequencia, existirao exercicios
# 
# Gerando x, que eh uma variavel aleatoria (pode ser qualquer historia) que for
# que tem 1 mil observacoes e segue uma normal (isso eh proposital)
x <- rnorm(1000)

# Teste de kolmogorov - Smirnov (KS)
# H0: amostra provem de uma populacao normal
# H1: amostra nao provem de uma populacao normal
ks.test (x, "pnorm", mean(x),sd(x))

# Teste de Shapiro Wilks
# H0: amostra provem de uma populacao normal
# H1: amostra nao provem de uma populacao normal
shapiro.test(x)

# sempre legal fazer um histogramazinho e dar aquela olhadinha
hist(x)

### Exercicio 
### Crio agora outra variavel aleatoria W. 
### Faca os testes estatisticos que aplicarem
W <- runif(3000)

# Teste de kolmogorov - Smirnov (KS)
# H0: amostra provem de uma populacao normal
# H1: amostra nao provem de uma populacao normal
ks.test (W, "pnorm", mean(W),sd(W))

# sempre legal fazer um histogramazinho e dar aquela olhadinha
hist(W)

# Teste de homogeneidade de variancia
# testa se variancias de k populacoes sao homogeneas
# Teste F de Levene 
 
# Para tal, geramos duas distribuicoes normais e faremos os testes
# de homogeneidade de variancia
# importar o dado variancia.xlsx
# e testar se as variancias sao iguais ou nao
 
# importando o arquivo
variancia <- read_excel("ASN-DSA-T5/11-TE/data/variancia.xlsx", 
                        col_types = c("numeric", "text"))

# Teste F de Levene
# H0: as variancias sao iguais
# H1: pelo menos 1 variancia e diferente
leveneTest(medida ~ grupo, data=variancia)

# para visualizar e ter a sensacao
variancia %>% 
  ggplot() + 
  geom_boxplot(aes(x=grupo, y=medida))

### EXERCICIO 
### importe o arquivo variancia2.xlsx
### Teste se as variancias sao iguais ou diferentes
### visualize isso atraves de um boxplot
variancia2 <- read_excel("ASN-DSA-T5/11-TE/data/variancia2.xlsx", 
                        col_types = c("numeric", "text"))

colnames(variancia2)

leveneTest(x ~ grupo, data=variancia2)

variancia2 %>% 
  ggplot() + 
  geom_boxplot(aes(x=grupo, y=x))

# Teste t para uma amostra
# H0: mu = m0 (mu e a nossa famosa media mi e m0 e o numero que se quer medir)
# H1: mu <> m0

# gerando dados propositais
gerando <- rnorm(1000, mean = 10, sd=2)

# quero saber se a media da populacao e igual a 10 ou diferente
# H0: mu = 10
# H1: mu <> 10
t.test(gerando, mu=10)

# quero saber se a media da populacao eh 10 ou maior que 10
# H0: mu = 10
# H1: mu > 10
t.test(gerando, mu=10, alternative = "greater")

# quero saber se a media da populacao eh 10 ou menor que 10
# H0: mu = 10
# H1: mu < 10
t.test(gerando, mu=10, alternative = "less")


# Teste t para comparar duas medias
# H0: muA = muB
# H1: muA <> muB
# Lembrando que para esse teste eh necessario ser normal (no grupo A e B) e aplicar o teste
# certo para cada situacao de homonegidade de variancia
 
# vamos fazer com os dados variancia2
variancia2

# testa normalidade para cada grupo
# pegando apenas grupo_A
g_a <- variancia2 %>% 
  filter(grupo=="grupo_A")
shapiro.test(g_a$x)

# pegando apenas grupo_B
g_b <- variancia2 %>% 
  filter(grupo=="grupo_B") 
shapiro.test(g_b$x)

# fazendo teste de variancia para saber se sao iguais ou nao
leveneTest(x ~ grupo, data=variancia2)

# como sao variancias iguais, devemos fazer o teste t de duas amostras para variancias iguais!!!
# entao agora estamos testando se as medias sao iguais
# H0: media_grupo_A = media_grupo_B
# H1: media_grupo_A <> media_grupo_B
t.test(x ~ grupo, data=variancia2, var.equal=TRUE)

# media_A maior que media_B?
# como sao variancias iguais, devemos fazer o teste t de duas amostras para variancias iguais!!!
# entao agora estamos testando se as medias sao iguais
# H0: media_grupo_A = media_grupo_B
# H1: media_grupo_A > media_grupo_B
t.test(x ~ grupo, data=variancia2, alternative="greater", var.equal=TRUE)

# media_A menor que media_B?
# como sao variancias iguais, devemos fazer o teste t de duas amostras para variancias iguais!!!
# entao agora estamos testando se as medias sao iguais
# H0: media_grupo_A = media_grupo_B
# H1: media_grupo_A < media_grupo_B
t.test(x ~ grupo, data=variancia2, alternative="less", var.equal=TRUE)

### EXERCICIO 
### O artigo A Critical Appraisal of 98.6 Degrees F, the Upper Limit of the Normal Body Temperature, 
### and Other Legacies of Carl Reinhold August Wunderlich?  questiona a nocao de que a verdadeira 
### temperatura corporal media eh 98,6. Alem disso ha tambem alguma duvida sobre se as temperaturas 
### corporais medias para as mulheres sao iguais as dos homens (nivel de significancia de 0,01). 
### 
### Os dados sao NormTemp.sas7bdat e o dicionario de dados:
### 
### ID - Identificador do individuo
### BodyTemp - Temperatura corporal (graus Fahrenheit)
### Gender - Sexo do individuo
### HeartRate - Frequencia cardiaca (batimentos por minuto)
### 
### Referencia
### https://jamanetwork.com/journals/jama/article-abstract/400116


# ANOVA de 1 fator
# importar o arquivo  
ANOVA_1fator <- read_excel("dados/ANOVA_1fator.xlsx")

# fazendo o boxplot por perfil_idade
ANOVA_1fator %>% 
  ggplot()+
  geom_boxplot(aes(x= Perfil_idade, y=Gasto))

#Fazendo a ANOVA direto no R
anova_1fator = aov(Gasto ~ Perfil_idade, data=ANOVA_1fator)
summary(anova_1fator)

# Lembrando que tem os mesmos pre supostos de normalidade e variancias homogeneas 
# (nada aqui foi testado por ser apenas um exemplo didatico com poucas linhas)

# ANOVA de 2 fatores 
# quando tenho mais do que um fator e quero verificar os efeitos!!!
df_anova_2fatores <- read_excel("dados/ANOVA_2fatores.xlsx", 
                                col_types = c("numeric", "text", "text"))
names(df_anova_2fatores)

#Fazendo a ANOVA direto no R
anova_2fatores = aov(gastos ~ sexo*renda, data=df_anova_2fatores)
summary(anova_2fatores)

