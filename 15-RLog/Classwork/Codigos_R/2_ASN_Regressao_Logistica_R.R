#########################
## Regressao Logistica

# Codigos necessarios para esse curso.
# Foco na interpretacao dos resultados!

#* Regressao Logistica
#* Eh uma generalizacao da Regressao Linear, por isso faz parte
#* da familia dos modelos lineares generalizados.
#* - gera a probabilidade do evento de interesse
#* - metodo de maxima verossimilhanca
#* - intepretacao pela odds ratio

# importando bibliotecas
library(haven)
library(tidyverse)
library(rcompanion) # cramersV
library(caret)
library(pROC)

#**** Contexto de Negocio
#**** Um navio naufragou e estamos tentando entender quais foram os criterios 
#**** utilizados por eles, para definir quem eram os seres que usariam os botes 
#**** disponiveis!
#**** Conseguem nos ajudar a entender esses fatores?
#**** dados: titanic.sas7bdat

#importando base titanic
titanic <- read_sas("dados/titanic.sas7bdat")
View(titanic)

#** O primeiro passo eh sempre interessante entender um pouco mais sobre os dados
#** e nada melhor do que comecar a cronstruir o nosso "mapa astral" dos dados. 
#** Vamos descreve-los e pensar no que precisamos nos preocupar em relacao a cada 
#** variavel?

#** Dicionario de dados:
#** Variável - Descricao - Pensamento
#** Name - Nome do passageiro - O nome nao ira nos ajudar no modelo, certo? serve apenas para identificacao!!
#** Age - Idade - Como sera a distribuicao dessa variavel? Tem missing?
#** Gender - Genero do passageiro - Quais sao as opcoes que existem nessa variavel? Como esta a frequencia para cada um dos niveis possiveis? Tem missing?
#** Class - Classe no navio - Qual a classe que o passageiro estava alocado? Quantos niveis existem?
#** Fare - Valor pago pela viagem - Como sera a distribuicao dessa variavel? Tem missing? Faz sentido com a variavel anterior?
#** Survival -  Marcacao de quem sobreviveu (1) e quem nao (0) - Existe missing? Qual a frequencia de cada nivel?


#*** Vamos começar explorando cada uma das variaveis?

#*** Pratica - Analise Univariada

#*** Observando missing nas variaveis
table(is.na (titanic$Survival))
table(is.na (titanic$Gender)) 
table(is.na (titanic$Class))
table(is.na (titanic$Age)) 
table(is.na (titanic$Fare)) 
# Calcular percentual de valores ausentes para cada variável
colMeans(is.na(titanic)) * 100


#*** Explorando a variavel Age
#*** descritivas
summary(titanic$Age)

ggplot(titanic, aes(x = Age)) +
  geom_histogram()

ggplot(titanic, aes(x = Age)) + 
  geom_boxplot(color="black", fill="blue")


#*** Explorando a variavel Gender
#*** descritivas
table(titanic$Gender)

ggplot(titanic, aes(x = Gender, fill = Gender)) +
  geom_bar()

prop.table(table(titanic$Gender)) * 100

#*** Explorando a variavel Class
#*** descritivas
titanic$Class <- as.character(titanic$Class)
  
table(titanic$Class)

ggplot(titanic, aes(x = Class)) +
  geom_bar()

prop.table(table(titanic$Class)) * 100

#*** Explorando a variavel Fare
#*** descritivas
summary(titanic$Fare)

ggplot(titanic, aes(x = Fare)) + 
  geom_boxplot(color="black", fill="blue")

ggplot(titanic, aes(x = Fare)) +
  geom_histogram()


#*** Explorando a variavel Survival
#*** descritivas
table(titanic$Survival)

ggplot(titanic, aes(x = Survival)) +
  geom_bar()

prop.table(table(titanic$Survival)) * 100

#** Podemos ajustar nosso dicionario, com nossos novos aprendizados
#** Atualizando o Dicionario:
#** Variável - Descricao - Comentario
#** Name - Nome do passageiro
#** Age - Idade - Tem 263 missingins. Distribuicao levemente assimetrica a direita. Tem relacao linear com y (logito de p)?
#** Gender - Gereno do passageiro - Nao existe missing, temos mais homens do que mulheres, nao tem classe com n muito pequeno.
#** Class - Classe no navio - Nao existe missing, transformei em caracter, sao 3 classes. A classe 3 tem  o dobro das outras.
#** Fare - Valor pago pela viagem - Tem 1 missing. Distribuicao bem assimetrica a direita. Tem relacao linear com y (logito de p)? 
#** Survival -  Marcacao de quem sobreviveu (1) e quem nao (0) - Nao tem missing. 61,80% de mortes (0) e 38,20% de sobreviventes (1)



#*** Vamos explorar a relacao entre as variaveis?
#*** Algumas perguntas que podemos responder:
#*** - Existe uma relacao linear entre Age e Fare, com o Y (logito de p)?
#*** - Se existe missing, o que fazer com eles?
#*** - Existe relacao entre Fare e Class? Sera que as medias sao diferentes? 
#*** (se forem, precisamos colocar as duas no modelo?)
#*** - Como eh a distribuicao do Gender, Age e Class por Survival?

#*** Pratica - Analise Bivariada

#*** Comecando pela variavel Age

#*** Essa variavel tem missing, vou tentar ver o comportamento, colocando eles 
#*como se fosse uma classe

#*** Relacao de Survavil com a variavel numerica Age
ggplot(titanic, aes(x = as.factor(Survival), y = Age)) + 
  geom_boxplot(color="red", fill="orange", alpha=0.2)+
  labs(x = "Survival", y = "Age")

#*** Verificando a linearidade entre Age e Survival 
#*** Criando uma copia para manipular o que importa
df_age_cm <- titanic %>%
  select(Survival, Age) %>%
  arrange(Age) %>%
  mutate(percentil_age = cut(
    row_number(),
    breaks = c(-Inf, (n() / 10), (n() / 10) * 2, (n() / 10) * 3, (n() / 10) * 4,
               (n() / 10) * 5, (n() / 10) * 6, (n() / 10) * 7, (n() / 10) * 8, 
               (n() / 10) * 9, Inf),
    labels = c('10', '20', '30', '40', '50', '60', '70', '80', '90', '100'),
    include.lowest = TRUE
  ))

#*** Verificando a taxa de resposta em cada percentual
age_sum_cm <- df_age_cm %>%
  group_by(percentil_age) %>%
  summarize(Age = max(Age), n = n(), Survival = mean(Survival),
            logito_p = log(Survival / (1 - Survival)) ) %>%
  ungroup()
age_sum_cm

#*** Fazendo o gráfico de dispersão no p
ggplot(age_sum_cm, aes(x = percentil_age, y = Survival)) +
  geom_point() +
  labs(x = "Percentil da Idade", y = "Survival")

#*** Fazendo o gráfico de dispersão no logito de p
ggplot(age_sum_cm, aes(x = percentil_age, y = logito_p)) +
  geom_point() +
  labs(x = "Percentil da Idade", y = "logito_p")
age_sum_cm

#*** Fazendo o gráfico de dispersão
ggplot(age_sum_cm, aes(x = percentil_age, y = Survival)) +
  geom_point() +
  labs(x = "Percentil da Idade", y = "Survival")
age_sum_cm

#*** Como visto, a taxa de resposta de ate 16 anos eh diferente das demais
#*** notamos tambem que os dados faltantes estao com taxa de resposta parecidas
#*** pensando nisso decisao de agrupar a idade em 3 grupos (variavel: Age)
#*** Menor_16, Maior_16, Sem_idade
titanic$Age_C <- cut(
  titanic$Age,
  breaks = c(-Inf, 17, Inf),
  labels = c('Menor_16', 'Maior_16'),
  ordered = FALSE
)
titanic$Age_C <- ifelse(is.na(titanic$Age_C), "Sem_idade", as.character(titanic$Age_C))

#*** Verificando a quantidade e taxa de resposta no grupo
table(titanic$Age_C)
ggplot(titanic, aes(x = Age_C, fill = Age_C)) +
  geom_bar() +
  labs(x = "Age", y = "Contagem") +
  theme(axis.text.x = element_text(angle = 0))

titanic %>% 
  group_by(Age_C) %>% 
  summarise(tx_resposta = mean(Survival))

#** Essa nova variavel foi criada, pois como pudermos ver, a variavel Age nao tinha 
#** uma relacao linear com a resposta (Survival). Decidimos categorizar em 3 grupos, 
#** visto o momento da taxa de resposta observada por percentis da idade. Escolheu-se 
#** por agrupar os "sem idade" (nan), pois tinham uma taxa diferente dos outros 2 grupos. 

#** Atualizando o dicionario de dados:
#** Variável - Descricao - Comentario
#** Name - Nome do passageiro
#** Age - Faixa de Idade - Nao tem missing, sao 3 grupos com taxas de respostas bem diferentes.
#** Gender - Gereno do passageiro - Nao existe missing, temos mais homens do que mulheres, nao tem classe com n muito pequeno.
#** Class - Classe no navio - Nao existe missing, transformei em caracter, sao 3 classes. A classe 3 tem  o dobro das outras.
#** Fare - Valor pago pela viagem - Tem 1 missing. Distribuicao bem assimetrica a direita. Tem relacao linear com y? 
#** Survival -  Marcacao de quem sobreviveu (1) e quem nao (0) - Nao tem missing. 61,80% de mortes (0) e 38,20% de sobreviventes (1)



#*** Voltando a analise
#*** Vamos explorar a relação entre as variáveis?
#*** Algumas perguntas que podemos responder:
#*** - Existe uma relação linear entre Age e Fare, com o Y?
#*** - Se existe missing, o que fazer com eles?

#*** Para a variavel Fare faremos a mesma analise

#*** Como a variavel Fare tem apenas um missing, e nos sabemos que eh do dono do 
#*** navio, decidimos imputar o valor 0, pois ele nao pagou nada 
titanic$Fare <- ifelse(is.na(titanic$Fare), 0, as.numeric(titanic$Fare))

#*** Relacao de Survavil com a variavel numerica Fare
ggplot(titanic, aes(x = as.factor(Survival), y = Fare)) + 
  geom_boxplot(color="red", fill="orange", alpha=0.2) +
  labs(x = "Survival", y = "Fare")

#*** Verificando a linearidade entre Fare e Survival (apos tratar os mising)
#*** Criando uma copia para manipular o que importa
df_fare_cm <- titanic %>%
  select(Survival, Fare) %>%
  arrange(Fare) %>%
  mutate(percentil_fare = cut(
    row_number(),
    breaks = c(-Inf, (n() / 10), (n() / 10) * 2, (n() / 10) * 3, (n() / 10) * 4,
               (n() / 10) * 5, (n() / 10) * 6, (n() / 10) * 7, (n() / 10) * 8, 
               (n() / 10) * 9, Inf),
    labels = c('10', '20', '30', '40', '50', '60', '70', '80', '90', '100'),
    include.lowest = TRUE
  ))

#*** Fazendo a mesma coisa de cima (por isso mantive o nome da tabela) de uma 
#*** forma bem mais simples, utilizando uma funcao pronta
df_fare_cm <- titanic %>%
  select(Survival, Fare) %>%
  arrange(Fare) %>%
  mutate(percentil_fare = ntile(Fare, 10)) 

#*** Verificando a taxa de resposta em cada percentual
fare_sum_cm <- df_fare_cm %>%
  group_by(percentil_fare) %>%
  summarize(Fare = min(Fare), n= n(), Survival = mean(Survival),
            logito_p = log(Survival / (1 - Survival)) ) %>%
  ungroup()
fare_sum_cm

#*** Fazendo o gráfico de dispersão no p
ggplot(fare_sum_cm, aes(x = percentil_fare, y = Survival)) +
  geom_point() +
  labs(x = "Percentil da Fare", y = "Survival")

#*** Fazendo o gráfico de dispersão no logito de p
ggplot(fare_sum_cm, aes(x = percentil_fare, y = logito_p)) +
  geom_point() +
  labs(x = "Percentil da Fare", y = "logito_p")
fare_sum_cm

#** Como podemos ver, essa sim tem uma relacao linear com o logito de p, ou seja, 
#** pode ser utilizada como numero no modelo!
#** No entanto, me surge a curiosidade de que Fare e a variavel Class sejam 
#** a mesma coisa!!!

#*** Vamos verificar como eh a distribuicao da Fare para cada uma das classes 
#*** (variavel: Class)
ggplot(titanic, aes(x = as.factor(Class), y = Fare)) + 
  geom_boxplot(color="blue", fill="green", alpha=0.2) +
  labs(x = "Class", y = "Fare")
#* Olhando o boxplot podemos ver que aparentemente as medias sao diferentes
#* vou fazer um histograma, para poder observar a distribuicao tambem
ggplot(titanic, aes(x = Fare, fill = factor(Class))) +
  geom_density(alpha = 0.5) +
  facet_wrap(~Class,ncol=1)

#* realmente parece que as medias sao diferentes
#* e lembre que se as medias sao diferentes eh porque existe algum tipo de associacao
#* vou fazer a anova e depois o teste t para verificar
#* mas antes preciso validar a normalidade (o que nao parece ser)
#* # Shapiro - teste de normalidade
#* H0: eh normal
#* H1: nao eh normal
shapiro.test(titanic$Fare[titanic$Class=="1"])
shapiro.test(titanic$Fare[titanic$Class=="2"])
shapiro.test(titanic$Fare[titanic$Class=="3"])
# como nao eh normal (nenhuma delas), nao poderiamos usar a ANOVA, entao
# vou usar o teste Kruskal Wallis
# H0: mediana_class1 = mediana_class2 = mediana_class3
# H1: pelo menos 1 diferente  
kruskal.test(Fare ~ Class, data = titanic)
# rejeitamos H0, ou seja, pelo menos uma eh diferente

# vamos verificar o teste de Mann-Whitney de comparacao de medias entre class 1 com a 2
# H0: mediana_class1 - mediana_class2 = 0
# H1: mediana_class1 - mediana_class2 <> 0
wilcox.test(titanic$Fare[titanic$Class=="1"], titanic$Fare[titanic$Class=="2"], paired=FALSE)
# rejeito H0, as medias sao diferentes

# vamos verificar o teste de Mann-Whitney de comparacao de medias entre class 1 com a 3
# H0: mediana_class1 - mediana_class3 = 0
# H1: mediana_class1 - mediana_class3 <> 0
wilcox.test(titanic$Fare[titanic$Class=="1"], titanic$Fare[titanic$Class=="3"], paired=FALSE)
# rejeito H0, as medias sao diferentes

# vamos verificar o teste de Mann-Whitney de comparacao de medias entre class 2 com a 3
# H0: mediana_class2 - mediana_class3 = 0
# H1: mediana_class2 - mediana_class3 <> 0
wilcox.test(titanic$Fare[titanic$Class=="2"], titanic$Fare[titanic$Class=="3"], paired=FALSE)
# rejeito H0, as medias sao diferentes

#** Como podemos ver, ela tem uma relacao linear com a resposta, podendo assim 
#** entrar no modelo como uma variavel numerica! No entanto, fica claro que a variavel 
#** Fare tem uma relacao direta com a variavel Class.
#** Sendo assim, devemos ficar atentos a isso na hora de criar o modelo. As duas 
#** juntas pode nao ser uma boa opcao!


#** Atualizando o dicionario de dados:
#** Variável - Descricao - Comentario
#** Name - Nome do passageiro
#** Age - Faixa de Idade - Nao tem missing, sao 3 grupos com taxas de respostas bem diferentes.
#** Gender - Gereno do passageiro - Nao existe missing, temos mais homens do que mulheres, nao tem classe com n muito pequeno.
#** Class - Classe no navio - Nao existe missing, transformei em caracter, sao 3 classes. A classe 3 tem  o dobro das outras.
#** Fare - Valor pago pela viagem -  Tem uma relacao linear com logito de p, mas nao deveria ser utilizada junto com Class
#** Survival -  Marcacao de quem sobreviveu (1) e quem nao (0) - Nao tem missing. 61,80% de mortes (0) e 38,20% de sobreviventes (1)


#*** Voltando a analise
#*** Vamos explorar a relacao entre as variaveis?
#*** Algumas perguntas que podemos responder:
#*** - Como eh a distribuicao do Gender, Age e Class por Survival?

#* Entendendo um pouco mais da relacao entre as variaveis categoricas:
#* Para visualizar comportamento das categoricas juntas, podemos estudar 
#* graficos de barras e a estatistica cramer´s V

#*** Para a variavel Class
ggplot(titanic, aes(x = factor(Class), fill = factor(Survival))) +
  geom_bar(position = "dodge", stat = "count") +
  labs(title = "Sobrevivência por Classe",
       x = "Classe",
       y = "Frequência",
       fill = "Sobrevivência")

#*** Calculando a relacao de Class com Survival atraves do Cramer´s V
class_qui <- table(titanic$Survival, titanic$Class)
chisq.test(class_qui)
cramerV(class_qui)

#** Resumo da variavel Class
#** Notamos que eh uma variavel que pode ser muito util visto que existem diferencas
#** bizarras entre 0 e 1´s em cada uma das classes.   
#** A correlacao com y é de 0,3125

#*** Para a variavel Gender
ggplot(titanic, aes(x = Gender, fill = factor(Survival))) +
  geom_bar(position = "dodge", stat = "count") +
  labs(title = "Sobrevivência por Gênero",
       x = "Gênero",
       y = "Frequência",
       fill = "Sobrevivência")

#*** Calculando a relacao de Class com Survival atraves do Cramer´s V
gender_qui <- table(titanic$Survival, titanic$Gender)
chisq.test(gender_qui)
cramerV(gender_qui)

#** Resumo da variavel Gender
#** Notamos que eh uma variavel que pode ser ate mais util visto que existem diferencas
#** bizarras entre 0 e 1´s em cada um dos generos.   
#** A correlacao com y é de 0,529

#*** Para a variavel Age_C
ggplot(titanic, aes(x = Age_C, fill = factor(Survival))) +
  geom_bar(position = "dodge", stat = "count") +
  labs(title = "Sobrevivência por Age_C",
       x = "Agre_C",
       y = "Frequência",
       fill = "Sobrevivência")

#*** Calculando a relacao de Class com Survival atraves do Cramer´s V
age_c_qui <- table(titanic$Survival, titanic$Age_C)
chisq.test(age_c_qui)
cramerV(age_c_qui)

#** Resumo da variavel Age_C
#** Notamos que eh uma variavel pouco util visto que nao existem diferencas
#** bizarras entre 0 e 1´s em cada uma das categorias de idade.   
#** A correlacao com y é de 0,1404 (repare que eh a mais baixa)

#*** Voltando a analise
#** Atualizando o dicionario de dados:
#** Variável - Descricao - Comentario
#** Name - Nome do passageiro
#** Age - Faixa de Idade - Nao tem missing, sao 3 grupos com taxas de respostas bem diferentes entre si. Mas os comportamentos em y sao muito parecidos entre as classes.
#** Gender - Gereno do passageiro - Variavel com 2 classes. O comportamento em y eh muito diferente por classe, sendo entao a mais relacionada com y
#** Class - Classe no navio - Variavel com 3 classes. E comportamento em y difere, eh util
#** Fare - Valor pago pela viagem -  nao vou usar visto a relacao com Class
#** Survival -  Marcacao de quem sobreviveu (1) e quem nao (0) - Nao tem missing. 61,80% de mortes (0) e 38,20% de sobreviventes (1)



#*** Agora temos as definicoes que nos importam.   
#*** Y bem definido e X criadas: Idade, Gender e Class
# Como sao 3 variaveis categoricas, precisamos criar as dummies

#*** Criando as dummies (manualmente)
titanic_dummie <-  titanic %>% 
  mutate(
    Sexo_Fem = ifelse(Gender == "Female", 1, 0),
    Sexo_Masc = ifelse(Gender == "Male", 1, 0),
    Class_1 = ifelse(as.character(Class) == "1", 1, 0),
    Class_2 = ifelse(as.character(Class) == "2", 1, 0),
    Class_3 = ifelse(as.character(Class) == "3", 1, 0),
    Menor_16 = ifelse(Age_C == "Menor_16", 1, 0),
    Maior_16 = ifelse(Age_C == "Maior_16", 1, 0),
    Sem_idade = ifelse(Age_C == "Sem_idade", 1, 0)
  ) %>% 
  select( -Gender, -Class, -Age, -Fare, -Age_C)

#** Antes de irmos direto para o modelo com tudo:
#** Quero que notem e sintam algumas coisas =]
#** Veja a media da variavel resposta
provando <- titanic %>%
  group_by(Class) %>%
  summarise(media_survival = mean(Survival))
 
provando

#** Agora vamos fazer um modelo com apenas a variavel Class (ou seja, como sao
#** 3 niveis, vamos colocar apenas 2 dummies) e vamos ver qual eh o p_chapeu desse
#** modelo
model_teste <- glm(Survival ~ Class_1 + Class_2, 
                   family = "binomial", data = titanic_dummie)
summary(model_teste)

exp(model_teste$coefficients)

#* Repare que nosso modelo foi criado e pela odds ratio podemos interpretar a relacao
#* de Class com viver ou morrer.
#* OddsRatio (Class_1 x Class_3) = 4.743296
#* Quem esta na Class_1 tem 3,74 vezes mais chance de sobreviver do que os da Class_3
#* ou quem esta na Class_1 tem 374% mais chance de sobreviver do que os da Class_3
#* OddsRatio (Class_2 x Class_3) = 2.197077
#* Quem esta na Class_2 tem 1,19 vezes mais chance de sobreviver do que os da Class_3
#* ou quem esta na Class_2 tem 119% mais chance de sobreviver do que os da Class_3
#* (repare que ele relativiza com quem voce nao colocou no modelo!!)
#* Mas como saber que isso eh verdade?   

#***   Calculando na mao...

#*** Calculando a probabilidade de sobrevivencia para a classe 1
p_class1 <- mean(titanic_dummie$Survival[titanic_dummie$Class_1 == 1])
#*** Calculando a chance de sobrevivencia para a classe 1
c_class1 <- p_class1 / (1 - p_class1)
cat("A chance da Classe 1 sobreviver eh:", c_class1, "\n")

#*** Calculando a probabilidade de sobrevivencia para a classe 2
p_class2 <- mean(titanic_dummie$Survival[titanic_dummie$Class_2 == 1])
#*** Calculando a chance de sobrevivencia para a classe 2
c_class2 <- p_class2 / (1 - p_class2)
cat("A chance da Classe 2 sobreviver eh:", c_class2, "\n")

#*** Calculando a probabilidade de sobrevivencia para a classe 3
p_class3 <- mean(titanic_dummie$Survival[titanic_dummie$Class_3 == 1])
#*** Calculando a chance de sobrevivencia para a classe 3
c_class3 <- p_class3 / (1 - p_class3)
cat("A chance da Classe 3 sobreviver eh:", c_class3, "\n")

#*** Calculando a odds ratio (OR) entre a Classe 1 e a Classe 3
OR1_3 <- c_class1 / c_class3
cat("A OR de Classe 1 contra Classe 3 eh:", OR1_3, "\n")

#*** Calculando a odds ratio (OR) entre a Classe 2 e a Classe 3
OR2_3 <- c_class2 / c_class3
cat("A OR de Classe 2 contra Classe 3 eh:", OR2_3, "\n")

#** Viu como tudo sempre bate? Eh so entender como funciona!!!   
#** Agora vamos "aplicar" o modelo para cada class e verificarmos o p_chapeu 
#** (para voces notarem que eh a mesma coisa que fazer a media de Y pela variavel 
#** em questao)

# relembrando o resultado do modelo (para pegar os parametros)
summary(model_teste)

#*** Aplicando manualmente pela formula
p_class1 = (exp(-1.0706 + 1.5567*1 + 0.7871*0)) / (1+ exp(-1.0706 + 1.5567*1 + 0.7871*0))
p_class2 = (exp(-1.0706 + 1.5567*0 + 0.7871*1)) / (1+ exp(-1.0706 + 1.5567*0 + 0.7871*1))
p_class3 = (exp(-1.0706 + 1.5567*0 + 0.7871*0)) / (1+ exp(-1.0706 + 1.5567*0 + 0.7871*0))

cat("P_class1:", p_class1, "P_class2:", p_class2, "P_class3:", p_class3, "\n")

#*** A plicando atraves do predict
dados <- data.frame(
  Class_1 = c(1, 0, 0),
  Class_2 = c(0, 1, 0),
  Class_fim = c("class1", "class2", "class3")
)
dados$p_chapeu <- predict(model_teste, newdata = dados, type = "response")
dados

#** Reparou que eh exatamente a mesma coisa que o "provando"?
provando

#** Isso que eh importante entender e sentir.   
#** Voce notou que o p_chapeu nada mais eh do que a frequencia de sobreviventes 
#** (taxa de resposta) por Class?
  
#** TODO o modelo eh feito em cima de medias!!! 
  
#** Agora que entendemos, vamos colocar todas as variaveis no modelo e verificarmos
#** o que ira acontecer.

#*** Modelo com todas as variaveis
colnames(titanic_dummie)

model_titanic <- glm(Survival ~ . -Name -Sexo_Masc -Class_3 -Sem_idade, 
                     family = "binomial", data=titanic_dummie)
summary(model_titanic)

#*** Calculando a exponencial do parâmetro (Odds ration)
exp(model_titanic$coefficients)

# Repare que Maior_16 deu nao significativo, o que quer dizer que, na presenca de todas
# essas variaveis, Age_C no nivel Maior_16 pode ser agrupado com Sem_idade, pois
# acabam com o mesmo comportamento!

#*** Fazendo a selecao das variaveis
model_step <- step(model_titanic, direction = "both")
summary(model_step)

#*** Calculando a exponencial do parametro depois da selecao
exp(model_step$coefficients)


#*** Agora é com vocês!!!
#*** Como é a interpretação dessas Odds Ratios?
#*** Vamos colocar a mao na massa e calcular as metricas de qualidade de ajuste?
#*** Calcule:
#*** - MSE
#*** - Precision / Recall no corte 0,5 
#*** - Lift nos primeiros 10%
#*** - curva ROC e AUC

#*** Metricas
titanic_metricas <- titanic_dummie
#*** Coluna 'y_chapeu' sao as previsões do modelo
titanic_metricas$y_chapeu <- predict(model_step, newdata = titanic_metricas, type = "response")
titanic_metricas <- titanic_metricas %>%
  select(Name, Survival, y_chapeu)
#*** Coluna 'decisao_05' com base na coluna 'y_chapeu'
titanic_metricas <- titanic_metricas %>%
  mutate(decisao_05 = ifelse(y_chapeu >= 0.5, 1, 0))
titanic_metricas


#--------------------------------------------------

#* Interpretando Odds Ratio

#* OR (female x male) = 12.13   
#* Mulheres tem 11,13 vezes mais chance de sobreviver do que homens   
#* Mulheres tem 1.113% mais chance de sobreviver do que homens   

#* OR (class_1 x class_3) = 6.35   
#* Class_1 tem 5,35 vezes mais chance de sobreviver do que Class_3   
#* Class_1 tem 535% mais chance de sobreviver do que Class_3

#* OR (class_2 x class_3) = 2.42   
#* Class_2 tem 1,42 vezes mais chance de sobreviver do que Class_3   
#* Class_2 tem 142% mais chance de sobreviver do que Class_3

#* OR (menor_16 x maior_16_ou_sem_idade) = 2.55   
#* Menor_16 tem 1,55 vezes mais chance de sobreviver do que Maior_16_ou_sem_idade  
#* Menor_16 tem 155% mais chance de sobreviver do que Maior_16_ou_sem_idade

#*** Calculando MSE
titanic_metricas$MSE <- (titanic_metricas$Survival - titanic_metricas$y_chapeu)^2
MSE_modelo <- mean(titanic_metricas$MSE)
cat("MSE do modelo:", MSE_modelo, "\n")

#*** Calculando o RMSE
RMSE_modelo <- sqrt(MSE_modelo)
cat("RMSE do modelo:", RMSE_modelo, "\n")

#*** Matriz de confusao
conf <- table(titanic_metricas$Survival, titanic_metricas$decisao_05)
conf
# veja como funciona a soma da primeira linha (tudo que era 0 de verdade)
sum(conf[1,])

#*** Calculando a precision
#* Precision: de todos que estou dizendo que sao 1, quantos acertei
precision_05 <- conf[2,2]/sum(conf[,2])
cat("Precisão no corte 0.5:", precision_05, "\n")

#*** Calculando o recall
#* Recall: de todos que sao 1, quantos estou pegando
recall_05 <- conf[2,2]/sum(conf[2,])
cat("Recall no corte 0.5:", recall_05, "\n")

#*** Lift
#*** Vamos pegar os primeiros 10%
#*** Selecionando as colunas
df_y_chapeu <- titanic_metricas %>%
  select(Survival, y_chapeu) %>%
  arrange(desc(y_chapeu)) 

#** Criando uma variavel com o n_total da base
n_total <- nrow(titanic_metricas)

#*** Criando a variavel percentil do y_chapeu, de 10 em 10 %
df_y_chapeu <- df_y_chapeu %>%
  mutate(percentil_ychapeu = cut(row_number(), 
                                 breaks = c(-Inf, (n_total/10), (n_total/10)*2, 
                                            (n_total/10)*3, (n_total/10)*4, 
                                            (n_total/10)*5, (n_total/10)*6, 
                                            (n_total/10)*7, (n_total/10)*8, 
                                            (n_total/10)*9, +Inf),
                                 labels = c('10', '20', '30', '40', '50', '60',
                                            '70', '80', '90', '100')))

#*** Verificando a taxa de resposta em cada percentual (ou seja, por faixa, nao eh acumulado)
y_chapeu_sum <- df_y_chapeu %>%
  group_by(percentil_ychapeu) %>%
  summarise(media_ychapeu = mean(y_chapeu),
            media_survival = mean(as.numeric(Survival))) %>%
  ungroup()

#*** Calculando o lift para cada percentual
y_chapeu_sum <- y_chapeu_sum %>%
  mutate(Lift = media_survival / mean(as.numeric(titanic_metricas$Survival)))
y_chapeu_sum


# Criar um objeto ROC
roc_curve <- roc(titanic_metricas$Survival, titanic_metricas$y_chapeu)

# Calcular a AUC
area_curva_roc <- auc(roc_curve)

# Plotar a curva ROC
plot(roc_curve, main = "Curva ROC", col = "blue", lwd = 2)
abline(h = 0, v = 1, lty = 2, col = "gray")  # Linha de referência
legend("bottomright", legend = paste("AUC =", round(area_curva_roc, 3)), col = "blue", lty = 1, cex = 0.8)
