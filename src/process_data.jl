using CSV
using Dates
using DataFrames

include(joinpath(pwd(), "src", "get_data.jl"))

# SRAG

srag2019 = CSV.read(joinpath(pwd(), "data", "srag_raw_2019.csv"), DataFrame)
srag2020 = CSV.read(joinpath(pwd(), "data", "srag_raw_2020.csv"), DataFrame)
srag2021 = CSV.read(joinpath(pwd(), "data", "srag_raw_2021.csv"), DataFrame)

# Different Columns this is the set diff
setdiff(names(srag2019), names(srag2020), names(srag2021))

# intersect means to concatenate only the common columns
srag = vcat(srag2019, srag2020, srag2021; cols=:intersect)

# Twitter
files_twitter = filter(endswith(r"twitter_raw_\d{4}.csv"), readdir(joinpath(pwd(), "data"); join=true))
tweets = mapreduce(x -> CSV.read(x, DataFrame), vcat, files_twitter)

# TODO: process_twitter

# TODO: process_srag
