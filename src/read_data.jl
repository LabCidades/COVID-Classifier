using CSV
using Dates
using DataFrames
using Plots

# não temos s09 s18 s51
tweets = CSV.read(joinpath(pwd(), "data", "covid_twitter_time_series.csv"), DataFrame)
transform!(tweets, :n => ByRow(Int); renamecols=false)

describe(tweets)
unique(tweets.symptoms) |> println

total_tweets = combine(groupby(tweets, :symptoms), :n => sum)

sort(total_tweets, :n_sum; rev=true)

n_21 = filter(row -> row.symptoms == "s21", tweets)
n_06 = filter(row -> row.symptoms == "s06", tweets)

plot(n_21.date, n_21.n;
     label="dor de cabeça",
     leg=:topleft,
     xlabel="Data",
     yformatter = (x -> string(x / 1_000) * "K"))
plot!(n_06.date, n_06.n;
      label="cansaço")
