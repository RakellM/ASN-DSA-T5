#install.packages("plyr")
library("plyr")

#install.packages("dplyr")
library("dplyr")


library(tidyverse)

#install.packages("lubridate")
library(lubridate)
#install.packages("nycflights13")
library(nycflights13)


# Referência: https://rpubs.com/misken/getting-started-R-group-by


############################################
############################################
# Step 1: Load data
# Base de dados referente a agendamento de cirurgias com as seguintes variáveis
    #SurgeryDate: data da realização da cirugia - dia, mês e ano                 
    #Service: tipo de cirugia - cardiologia, ortopedia,  otorrino, gastro
    #ScheduledDaysInAdvance:Dias programados com antecedência
    #Urgency: se a cirurgia será de rotina ou de urgência 
    #InsuranceStatus: Particular, Plano de Saude, "tipo SUS", Nenhum

sched_df  <- read.csv("C://Users/crisr/Documents/ASN_Rocks/Series_Temporais/2. Curso_Completo/Dados/SchedDaysAdv.csv", stringsAsFactors = TRUE) 
# String ARE factors by default, just being explicit

head(sched_df,10)  # See the start of the data frame
tail(sched_df)  # See the end of the data frame

# Create variables day, month and year
sched_df$day <- format(as.Date(sched_df$SurgeryDate, format="%Y-%m-%d"),"%d")
sched_df$month <- format(as.Date(sched_df$SurgeryDate, format="%Y-%m-%d"),"%m")
sched_df$year <- format(as.Date(sched_df$SurgeryDate, format="%Y-%m-%d"),"%y")
head(sched_df)


############################################
############################################
# Step2: Using plyr for group wise analysis


# Agrupando a variável ScheduledDaysInAdvance para ter uma visão geral de como está
# a média, desvio padrão, min, percentil 5, percentil 95 e maximo
summarise(sched_df, mean_leadtime=mean(ScheduledDaysInAdvance),
          sd_leadtime=sd(ScheduledDaysInAdvance),
          min_leadtime = min(ScheduledDaysInAdvance),
          p05_leadtime = quantile(ScheduledDaysInAdvance,0.05),
          p95_leadtime = quantile(ScheduledDaysInAdvance,0.95),
          max_leadtime = max(ScheduledDaysInAdvance))
# O comando não é muito util para nossas agregações, mas é só para mostrar como fazer uma 
#agregação básica de um campo em um data frame


# armazenando os resultados do summarise em um data frame
overall_stats <- summarise(sched_df, mean_leadtime=mean(ScheduledDaysInAdvance),
                             sd_leadtime=sd(ScheduledDaysInAdvance),
                             min_leadtime = min(ScheduledDaysInAdvance),
                             p05_leadtime = quantile(ScheduledDaysInAdvance,0.05),
                             p95_leadtime = quantile(ScheduledDaysInAdvance,0.95),
                             max_leadtime = max(ScheduledDaysInAdvance))



## A variant of above but using the special "dot" function so that the splitting variables can
## be referenced directly by name without quotes.
ddply(sched_df,.(Urgency),summarise,numcases=length(ScheduledDaysInAdvance))

## Mean(ScheduledDaysInAdvance) by Urgency
ddply(sched_df,.(Urgency),summarise,mean_leadtime=mean(ScheduledDaysInAdvance))

## Mean(ScheduledDaysInAdvance) by Urgency and store result in an array
meansbyurg<-ddply(sched_df,.(Urgency),summarise,mean_leadtime=mean(ScheduledDaysInAdvance))

## Std(ScheduledDaysInAdvance) by Urgency
ddply(sched_df,.(Urgency),summarise,sd_leadtime=sd(ScheduledDaysInAdvance))


## Now let's do mean lead time by Urgency and InsuranceStatus
ddply(sched_df,.(Urgency,InsuranceStatus),summarise,mean_leadtime=mean(ScheduledDaysInAdvance))


# Let's compute the 95th percentile of lead time by service and insurance status.
ddply(sched_df,.(Service,InsuranceStatus),summarise,p95_leadtime=quantile(ScheduledDaysInAdvance,0.95))



############################################
############################################
# Step3: Using plyr for group wise dates

# 3.1 Sumariza a variavel ScheduledDaysInAdvance considerando
# Intervalo de Tempo: dia
# Método de acumulação: min, max, média, total
sched_df_day <- ddply(sched_df,.(year,month,day),summarise,
                      min_leadtime=min(ScheduledDaysInAdvance),
                      max_leadtime=max(ScheduledDaysInAdvance),
                      mean_leadtime=mean(ScheduledDaysInAdvance),
                      sum_leadtime=sum(ScheduledDaysInAdvance))
head(sched_df_day)

par(mfrow=c(4,1))
#plot(sched_df_day$min_leadtime)
plot(ts(sched_df_day$min_leadtime))

#plot(sched_df_day$max_leadtime)
plot(ts(sched_df_day$max_leadtime))

#plot(sched_df_day$mean_leadtime)
plot(ts(sched_df_day$mean_leadtime))

#plot(sched_df_day$sum_leadtime)
plot(ts(sched_df_day$sum_leadtime))


# vamos plotar uma parte dos dados  para verificar se conseguimos encontrar algum padrão
par(mfrow=c(1,1))
plot(ts((sched_df_day$sum_leadtime)[sched_df_day$year < 2008 & sched_df_day$month < 6]))



# 3.2 Sumariza a variavel ScheduledDaysInAdvance considerando
# Intervalo de Tempo: mês
# Método de acumulação: min, max, média, total
sched_df_month <- ddply(sched_df,.(year,month),summarise,
                      min_leadtime=min(ScheduledDaysInAdvance),
                      max_leadtime=max(ScheduledDaysInAdvance),
                      mean_leadtime=mean(ScheduledDaysInAdvance),
                      sum_leadtime=sum(ScheduledDaysInAdvance))
head(sched_df_month)

par(mfrow=c(4,1))
#plot(sched_df_month$min_leadtime)
plot(ts(sched_df_month$min_leadtime))

#plot(sched_df_month$max_leadtime)
plot(ts(sched_df_month$max_leadtime))

#plot(sched_df_month$mean_leadtime)
plot(ts(sched_df_month$mean_leadtime))
     
#plot(sched_df_month$sum_leadtime)
plot(ts(sched_df_month$sum_leadtime))
par(mfrow=c(1,1))



# 3.3 Sumariza a variavel ScheduledDaysInAdvance considerando
# Intervalo de Tempo: ano
# Método de acumulação: min, max, média, total
sched_df_year <- ddply(sched_df,.(year),summarise,
                        min_leadtime=min(ScheduledDaysInAdvance),
                        max_leadtime=max(ScheduledDaysInAdvance),
                        mean_leadtime=mean(ScheduledDaysInAdvance),
                        sum_leadtime=sum(ScheduledDaysInAdvance))
head(sched_df_year)

par(mfrow=c(4,1))
#plot(sched_df_year$min_leadtime)
plot(ts(sched_df_year$min_leadtime))

#plot(sched_df_year$max_leadtime)
plot(ts(sched_df_year$max_leadtime))

#plot(sched_df_year$mean_leadtime)
plot(ts(sched_df_year$mean_leadtime))

#plot(sched_df_year$sum_leadtime)
plot(ts(sched_df_year$sum_leadtime))
par(mfrow=c(1,1))
