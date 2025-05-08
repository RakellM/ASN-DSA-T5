# Testes estatisticos

library(readxl)
library(haven)
library(tidyverse)
library(car) #teste F de levene
library(RVAideMemoire) #teste q de cochran

# Legendas:
#  Explicacao Conteudo
## Dicas
### Pratica
#### Contexto Negocio

# Testes nao parametricos
# as hipoteses sao formuladas sobre caracteristicas qualitativas da populacao
# estes sao bem menos robustos que os parametricos (mas quando nao se tem cao, caca-se com gato!!!
# ou seja, perde-se um pouco, mas conclui-se algo!! - analogia da vela)
# sao testes livres de distribuicao (sem regras)

# Teste para uma amostra
#         . Teste binomial (binario)
#         . Teste qui-quadrado (nominal ou ordinal)
        
# Teste para 2 amostras emparelhadas
# emparelhadas significa "antes e depois"
#         . Teste McNemar (binario)       
#         . Teste Wilcoxon (ordinal)
                      
# Teste para 2 amostras independentes
#         . Teste qui-quadrado (nominal ou ordinal)     
#         . Teste dos Mann-Whitney (ordinal)
                         
# Teste para k amostras emparelhadas
# emparelhadas significa antes e depois
#         . Teste Q de Cochran (binario)       
#         . Teste de Friedman (ordinal)
       
# Teste para k amostras independentes
#         . Teste qui-quadrado (nominal ou ordinal)        
#         . Teste de Kruskal Wallis (ordinal)         


#  Teste binomial (binario) para uma amostra    
 
#### Contexto Negocio
#### ASN.Rocks aplicou 2 metodos de ensino (metodo A e B) 
#### e no final o aluno escolheu o melhor.
#### ASN.Rocks espera que nao exista diferenca entre os metodos
#### Imaginando o experimento, assuma que dos 30 alunos
#### 12 escolheram metodo A
#### 18 escolheram metodo B

# Neste caso, podemos fazer um teste binomial, pois a resposta eh 
# binaria: metodo A e B  
# H0: p = 0,5
# H1: p <> 0,5
binom.test(12, 30, p = 0.5, conf.level = 0.95)  

### Exercicio
### Em uma amostra de tamanho 50 foram observados 20 sucessos. 
### Testar se p>0.2 com nivel de significância de 5%.
### H0: p = 0,2
### H1: p > 0,2


# Teste qui-quadrado (nominal ou ordinal) para uma amostra 

#### Contexto Negocio
#### A Dri sempre quis saber se o numero de recem nascidos muda conforme o mes
# Neste caso estamos falando de meses, testar se a frequencia de nascimentos 
# eh parecida com a media geral nos diria que as frequencias sao as mesmas
# H0: nao existe diferenca significativa entre as frequencias observadas e as esperadas
# H1: existe diferenca      

# carregando valores observados      
nascimentos <- c(jan=320, fev=400, mar=200, abr=250, mai=330, jun=200, jul=200,
                 ago=250, set=320, out=400, nov=890, dez=780) 
# fazendo teste qui-quadrado
chisq.test(nascimentos)

### Exercicio
### Senhor Prezz eh dono de uma indústria que vende seu produto em 3 cidades 
### do seu estado. E, determinado dia, se questionou se o volume das vendas 
### eh o mesmo em todas as cidades ou se existem diferenças significativas.   
### Vamos ajudá-lo? Ele nos disse que na cidade de Betim, vendeu 450 unidades.
### Em Mariana vendeu 480 e em Diamantina 490 unidades.




# Teste McNemar (binario) para 2 amostras emparelhadas
# H0: As probabilidades das celulas [i,j] e [j,i] sao iguais (nao faz diferenca)
# H1: As probabilidades das celulas [i,j] e [j,i] sao diferentes (faz diferenca)

#### Contexto Negocio
#### Antes da pandemia a ASN.Rocks tinha um escritorio e turmas presenciais. 
#### Dado o coronga, essa turmas mudaram para online!
#### Toda aula a ASN.Rocks tem costume de perguntar se o aluno gosta ou 
#### nao gosta da mesma
#### Isso foi feito antes e depois do coronga, ou seja

# Repare que eh a mesma unidade amostral (aluno), 
# que responde a questão em dois momentos diferentes (emparelhado)
# So existem duas opcoes de escolha (Gosto ou Nao Gosto) - Binario
# O foco esta em saber se existiu mudanca apos algo 
# (nesse caso, mudança para aula online)
# Teste MCNemar

# vamos iniciar importando o arquivo com a informacoes
coronga <- read_excel("dados/coronga.xlsx")
# aplicando Teste MCNemar
mcnemar.test(coronga$Opiniao_Antes, coronga$Opiniao_Depois)
# ou seja, nao fez diferenca nenhuma!


# Teste Wilcoxon (ordinal) para 2 amostras emparelhadas
# este teste pode ser uma alternativa do teste t duas amostras, quando nao sao normais!
# repare que os dados quantitativos sao transformados em ordinais, logo, perca de poder!!!
# No entanto este eh mais poderoso que o teste dos sinais, pois leva em consideracao
# alem da direcao das diferencas para cada par, a magnitude da diferenca dentro de cada par!
# H0: mediana_antes - mediana_depois = 0
# H1: mediana_antes - mediana_depois <> 0

#### Contexto Negocio
#### Os alunos da ASN.Rocks foram submetidos a um teste antes do curso
#### e apos o curso, para verificar se o curso acrescentou valor.
#### ou seja, se os alunos melhoraram o desempenho na nota do teste.
# Repare que eh a mesma unidade amostral (aluno), em que a nota foi avaliada (emparelhado)
# Eh uma variavel quantitativa, a melhor opcao seria teste parametrico
# Importante testar normalidade, se nao for, entao:
# Teste Wilcoxon
# H0: mediana_antes - mediana_depois = 0
# H1: mediana_antes - mediana_depois <> 0

# comecamos importando o arquivo
notas_alunos <- read_excel("dados/notas_alunos_antes_depois.xlsx", 
                           col_types = c("numeric", "numeric"))
notas_alunos

# verificando normalidade das notas para cada amostra
shapiro.test(notas_alunos$notas_antes)
shapiro.test(notas_alunos$notas_depois)
# como podemos ver, nao temos normalidade nos dados, sendo assim
# H0: mediana_antes - mediana_depois = 0
# H1: mediana_antes - mediana_depois < 0
# H0: mediana_antes = mediana_depois 
# H1: mediana_antes < mediana_depois 

wilcox.test(notas_alunos$notas_antes, notas_alunos$notas_depois, paired=TRUE,
            alternative = c("less"))
# como p-valor menor que nivel de significancia, entao rejeito H0, ou seja,
# o curso mostrou ser util para melhorar o desempenho dos alunos


# Teste qui-quadrado (nominal ou ordinal) para 2 amostras independentes
# teste que tem a ideia de comparar com media geral, mas lembre-se que
# agora sao duas informacoes se cruzando!

#### Contexto Negocio
#### Sera que o clima (sol ou chuva) inlfuencia no humor (triste ou feliz) 
#### do meu chefe???

# Repare que sao duas variaveis categoricas
# Sao independentes
# Teste qui-quadrado
# H0: nao existe diferenca significativa entre as frequencias observadas e as esperadas
# H1: existe diferenca  
# # o arquivo chefe_humor_clima.xlsx contem os dados das experiencias passadas

# lendo arquivo
chefe <- read_excel("dados/chefe_humor_clima.xlsx", 
                    col_types = c("text", "text"))
names(chefe)
# realizando teste qui-quadrado
chisq.test(chefe$clima, chefe$humor)

# tambem podemos adicionar uma tabela cruzada no teste qui-quadrado
chefe_sumarizado<- table(chefe$clima, chefe$humor)
chisq.test(chefe_sumarizado)
# como p-valor menor que alpha, entao rejeito H0 e concluo que 
# existe alguma relacao do clima com o humor do meu chefe

### Exercicio
### Apos uma determinada campanha de marketing, oferecemos nosso produto.
### Sera que o genero (Fem / Masc) influencia na compra ou nao compra do 
### nosso produto?
### arquivo: compra.xlsx


# Teste dos Mann-Whitney (ordinal) para 2 amostras independentes
# um dos testes mais poderoros aplicado em variaveis quantitativas e qualitativas
# em escala ordinal. Eh o teste de wilcoxon para amostras independentes
# este teste pode ser uma alternativa do teste t duas amostras, quando nao sao normais!
# H0: mediana_A - mediana_B = 0
# H1: mediana_A - mediana_B <> 0

#### Contexto Negocio
#### Um medico testou um remedio para aumentar a tiroxina no sangue
#### Fez-se um experimento, com duas amostras: uma que tomou o placebo
#### outra que tomou o remedio
#### arquivo: tiroxina.xlsx

# nessa historia, como tiroxina eh uma mensuracao
# deveriamos comecar pelo teste t parametrico
# entao vamos testar normalidade primeiro
tiroxina <- read_excel("dados/tiroxina.xlsx", 
                       col_types = c("numeric", "numeric"))
# validando se nao da pra fazer o teste t 
shapiro.test(tiroxina$placebo)
shapiro.test(tiroxina$remedio)
# como os dados nao sao normais, entao o teste de Mann-Withney
# H0: mediana_remedio - mediana_placebo = 0
# H1: mediana_remedio - mediana_placebo > 0
# ou seja
# H0: mediana_remedio = mediana_placebo
# H1: mediana_remedio > mediana_placebo 
wilcox.test(tiroxina$remedio, tiroxina$placebo, paired=FALSE, alternative = c("greater"))
# como p-valor muito pequeno, entao rejeito H0 e assume-se 
# que o remedio realmente tem efeito no numero de tiroxina no sangue


# Teste Q de Cochran (binario) para k amostras emparelhadas
# extensao do teste McNemar
# testa se as frequencias ou proporcoes de 3 ou mais grupos emparelhados
# sao significativamente diferentes entre si
# H0: p1=p2=...=pk
# H1: pelo menos 1 diferente
# 
#### Contexto Negocio
#### Dri Gatinha ministra um curso de extensao sobre ciencia de dados
#### Apos o curso, ela examina seus alunos sobre a vontade de adotar boas praticas nas etapas: 
#### Entendimento_negocio, Criacao_ABT, Machine_Learning e Producao. 
#### Ela quer saber qual pratica os alunos estavam mais dispostos a adotar.

# este eh um teste emparelhado, pois eh o mesmo aluno respondendo varias questoes
# como a resposta eh binaria, entao o teste aplica
escola <- read_excel("dados/escola.xlsx", 
                     col_types = c("text", "text", "text"))
# teste q de cochran
cochran.qtest(Resposta ~ Pratica | Aluno,
              data = escola)
# como p-valor menor que alpha, entao rejeitamos H0, ou seja
# as proporcoes de adesao as boas praticas nao sao iguais nos temas pesquisados


# Teste de Friedman (ordinal) para k amostras emparelhadas
# extensao do teste wilcoxon para 3 ou mais amostras emparelhadas
# boa alternativa para ANOVA quando as suposicoes forem violadas 
# H0: Md1 = Md2 = ... = Mdk
# H1: pelo menos 1 diferente

#### Contexto Negocio
#### Um medico esta testando um tratamento de emagrecimento
#### Para tal, mediu o peso dos pacientes antes, durante e 
#### apos 3 meses de tratamento 

# Repare que sao dados quantitativos, pesos, ou seja, devemos comecar 
# tentando um teste parametrico (neste caso uma ANOVA de medida repetidas - emparelhada)
# Caso nao seja possível, entao utilizaremos o teste de Friedman
# Aqui vale um comentario: como dito varias vezes em sala de aula, 
# sempre existem consideracoes. 
# Exemplo: existem artigos que mostram que usar uma ANOVA para grandes volumes de dados 
# mesmo sem ser normal, você encontrara respostas confiáveis. No entanto para 
# quantidade pequenas (como desse exemplo com apenas 15 alunos), 
# os testes neo parametricos acabam sendo bem recomendados.
# H0: Md_antes = Md_durante = Md_depois
# H1: pelo menos 1 diferente

# importando a base de dados
tratamentos <- read_excel("dados/tratamentos2.xlsx", 
                          col_types = c("text", "text", "numeric"))
names(tratamentos)

friedman.test(peso ~ momento | ID, data = tratamentos)
# como p-valor muito pequeno, rejeitamos H0 entao as medianas sao diferentes
# o tratamento parece que funciona
# vamos plotar um boxplot para visualizar
tratamentos %>% 
  ggplot()+
  geom_boxplot(aes(x= peso, y=momento))


# Teste qui-quadrado (nominal ou ordinal) para k amostras independentes
# Eh o mesmo que ja conhecemos, mas agora com k amostras

#### Contexto Negocio
#### Sera que o produtividade (baixa, media, alta) varia 
#### conforme o Turno (T1, T2, T3 e T4)???

# Repare que sao duas variaveis categoricas com mais de 3 niveis cada
# Sao independentes
# Teste qui-quadrado
# O codigo eh o mesmo ja utilizado
# H0: nao existe diferenca significativa entre as frequencias observadas e as esperadas (nao tem diferenca por turno)
# H1: existe diferenca  (tem diferenca por turno)

# importando o arquivo onde cada linha foi uma medicao realizada no historico dos dias
produtividade <- read_excel("dados/produtividade.xlsx")
# realizando o teste qui-quadrado
chisq.test(produtividade$Produtividade, produtividade$Turno)


# Teste de Kruskal Wallis (ordinal)  para k amostras independentes
#  alternativa a ANOVA quando nao tem normalidade e homonegeidade da variancia
#  ou quando a variavel eh ordinal
#  Para k=2 Kruskal-Wallis eh o mesmo do Mann-Whitney
# H0: mediana_A = mediana_B = mediana_C
# H1: pelo menos 1 diferente   
# kruskal.test(medida ~ grupo, data = dataframe)

#### Contexto Negocio
#### Foi realizado um experimento para verificar desempenho de 
#### tratamentos diferentes em aumentar o peso das plantas secas
#### arquivo: Peso_Planta.xlsx

### Exercicio
### Responda a pergunta de negocio utilizando o teste ADEQUADO
# Aqui vale um comentario je feito: como dito varias vezes em sala de aula, 
# sempre existem consideracoes. 
# Exemplo: existem artigos que mostram que usar uma ANOVA para grandes volumes de dados 
# mesmo sem ser normal, você encontrara respostas confiáveis. No entanto para 
# quantidade pequenas (como desse exemplo com apenas 10 coletas), 
# os testes nao parametricos acabam sendo bem recomendados, mesmo que se 
# pelo estatistico parecam normais.

Peso_Planta <- read_excel("dados/Peso_Planta.xlsx")

# testando normalidade das amostras
ctrl_1 <- Peso_Planta %>% 
  filter(tratamento == "ctrl")
shapiro.test(ctrl_1$peso)
# aparentemente nao eh muito normal, ne?
hist(ctrl_1$peso)

# testando normalidade das amostras
trt1_1 <- Peso_Planta %>% 
  filter(tratamento == "trt1")
shapiro.test(trt1_1$peso)
# aparentemente nao eh muito normal, ne?
hist(trt1_1$peso)

# testando normalidade das amostras
trt2_1 <- Peso_Planta %>% 
  filter(tratamento == "trt2")
shapiro.test(trt2_1$peso)
# aparentemente nao eh muito normal, ne?
hist(trt2_1$peso)

# eu nao me sinto seguro para assumir essas distribuicoes como normais
# pouco dados e o histograma nao me agrada. Decidi pelo teste nao parametrico
kruskal.test(peso ~ tratamento, data = Peso_Planta)
# como p-valor maior que o nivel de significancia, entao nao rejeitamos h0
# ou seja, as medias de pesos por tratamento sao iguais

