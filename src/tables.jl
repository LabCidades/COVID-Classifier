using PrettyTables

include(joinpath(pwd(), "src", "read_data.jl"))

unique(tweets.symptoms) |> println

# Sintomas totais
total_tweets = combine(groupby(tweets, :symptoms_detail), :n => sum)
sort!(total_tweets, :n_sum; rev=true)
rename!(total_tweets, :symptoms_detail => :Sintomas, :n_sum => :Tweets)
top_15_tweets = total_tweets[1:15, :]

formatter = (v, i, j) -> j == 2 ? string(round(v / 1_000_000; digits=3)) * " mi" : j

pretty_table(top_15_tweets;
             nosubheader=true,
             title="Top-15 sintomas mais tuítados",
             formatters=formatter)
