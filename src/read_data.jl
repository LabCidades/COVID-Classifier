using AlgebraOfGraphics
using CairoMakie
using CSV
using Dates
using DataFrames

# no Twitter não temos s09 s18 s51
tweets = CSV.read(joinpath(pwd(), "data", "covid_twitter_time_series.csv"), DataFrame)
transform!(tweets, :n => ByRow(Int); renamecols=false)

describe(tweets)
unique(tweets.symptoms) |> println

# Sintomas totais
total_tweets = combine(groupby(tweets, :symptoms), :n => sum)
sort(total_tweets, :n_sum; rev=true)


# Pure Makie Implementation
# still need to figure out color palette
xs = min(tweets.date...):Day(1):max(tweets.date...)
months = min(tweets.date...):Month(1):max(tweets.date...)
lentime = length(xs)
slice_dates = 1:90:lentime
y_dor_de_cabeca = filter(row -> row.symptoms == "s21", tweets).n
y_cansaco = filter(row -> row.symptoms =="s06", tweets).n

# Gráfico
resolution = (800, 600)
f = Figure(; resolution)
ax = Axis(f[1,1];
          xlabel = "Data",
          ylabel = "Frequência em milhares")
l1 = lines!(ax, 1:length(xs), y_cansaco; color=:blue, label="cansaço")
l2 = lines!(ax, 1:length(xs), y_dor_de_cabeca; color=:red, label="dor de cabeca")
ax.xticks = (slice_dates, Dates.format.(xs, dateformat"mm-yy")[slice_dates])
ax.xticklabelrotation = π/4
ax.xlabelpadding = 4.0
axislegend("Sintoma"; position=:ct, orientation=:horizontal)
f

# AoG Implementation
# still need to figure out how to have more xticks

# Somente s21 e s06
n_21_06 = filter(row -> row.symptoms == "s21" || row.symptoms == "s06", tweets)

# Gráfico
resolution = (800, 600)
f = Figure(; resolution)
n_21_06 = filter(row -> row.symptoms == "s21" || row.symptoms == "s06", tweets)
plt = data(n_21_06) *
        mapping(:date=>"Data",
                :n=>(x -> x / 1_000) => "Frequência em milhares",
                color=:symptoms => renamer("s21" => "dor de cabeça", "s06" => "cansaço") => "Sintomas") *
        visual(Lines)

# dateticks = min(n_21_06.date...):Month(6):max(n_21_06.date...) |> collect
ag = draw!(f, plt;
           axis=(;
           title = "Frequência de sintomas no Twitter",
           titlesize=20,
           xticklabelrotation=π/4,
           xlabelpadding=4.0
           ))
legend!(f[end+1, 1], ag; orientation=:horizontal, tellheight=true, tellwidth=false)
f
