devtools::install_github("FinYang/tsdl")
library(tsdl)
library(forecast)
library(readr)

nomes <- tsdl::meta_tsdl$description

ts_to_df <- function(ts){
  return(data.frame(date = time(ts), valor = as.matrix(ts)))
}

# Série sunspot
sunspot <- subset(tsdl, description="Sunspot")

autoplot(sunspot[[1]])

sunspot[[1]]

sunspot_df <- ts_to_df(sunspot[[1]])
sunspot_df %>% 
  write_csv('/Users/filipeduarte/Desktop/Doutorado UFPE/2020.1/Swarm Intelligence/Artigo/sunspot.csv')

# Série International airline passengers time series
airline_passengers <- subset(tsdl, description = "airline")

autoplot(airline_passengers[[1]])

airline_passengers_df <- ts_to_df(airline_passengers[[1]])

airline_passengers_df %>% 
  write_csv('/Users/filipeduarte/Desktop/Doutorado UFPE/2020.1/Swarm Intelligence/Artigo/airline_passengers.csv')

# Série Australian wine sales time series
wine_sales <- subset(tsdl, description = "wine")

autoplot(wine_sales[[1]][,"Drywhite"])

wine_sales_df <- ts_to_df(wine_sales[[1]][, "Drywhite"])

wine_sales_df %>% 
  write_csv("/Users/filipeduarte/Desktop/Doutorado UFPE/2020.1/Swarm Intelligence/Artigo/wine_sales.csv")

# Série S&P 500 
sp500 <- subset(tsdl, description = "S&P 500")

autoplot(sp500[[2]])

sp500_df <- ts_to_df(sp500[[2]])

sp500_df %>% 
  write_csv("/Users/filipeduarte/Desktop/Doutorado UFPE/2020.1/Swarm Intelligence/Artigo/sp500.csv")

# Série US Death 
usa_death <- subset(tsdl, description = "Accidental deaths in USA")

autoplot(usa_death[[1]])

usa_death_df <- ts_to_df(usa_death[[1]])

usa_death_df %>% 
  write_csv("/Users/filipeduarte/Desktop/Doutorado UFPE/2020.1/Swarm Intelligence/Artigo/usa_accident_death.csv")

# Série Internet Traffic 
internet_traffic <- subset(tsdl, description = "Internet traffic")

autoplot(internet_traffic[[6]])

internet_traffic_df <- ts_to_df(internet_traffic[[6]])

internet_traffic_df %>% 
  write_csv("/Users/filipeduarte/Desktop/Doutorado UFPE/2020.1/Swarm Intelligence/Artigo/internet_traffic.csv")

# Série Daily minimum temperature 
daily_min_temperature <- subset(tsdl, description = "Daily minimum")

autoplot(daily_min_temperature[[1]])

daily_min_temperature_df <- ts_to_df(daily_min_temperature[[1]])

daily_min_temperature_df %>% 
  write_csv('/Users/filipeduarte/Desktop/Doutorado UFPE/2020.1/Swarm Intelligence/Artigo/daily_temp.csv')
