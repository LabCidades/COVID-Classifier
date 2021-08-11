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
function process_twitter()
    files_twitter = filter(endswith(r"twitter_raw_\d{4}.csv"), readdir(joinpath(pwd(), "data"); join=true))
    tweets = mapreduce(x -> CSV.read(x, DataFrame), vcat, files_twitter)
    select!(tweets, :id, :date, :tweet, :symptoms, :nsymptoms)
    transform!(tweets,
            :date => ByRow(x -> Date(x, dateformat"Y-m-d H:M:S")),
            :symptoms => ByRow(x -> split(x, ", "));
            renamecols=false)
    tweets = flatten(tweets, :symptoms)
    # Create a dataframe for all dates and symptoms
    df_dates = DataFrame(date=Date("2019-01-01"):Day(1):Date("2021-06-30"))
    df_symptoms = DataFrame(symptom=unique(tweets.symptoms))
    df_union = crossjoin(df_dates, df_symptoms)
    # combine the tweets df by date and symptom and count row as n
    df_count = combine(groupby(tweets, [:date, :symptoms]), nrow => :n)
    df_final = leftjoin(df_union, df_count, on=[:date, :symptom => :symptoms])
    transform!(df_final, :n => ByRow(x -> coalesce(x, 0)); renamecols=false)
    sort!(df_final, [:date, :symptom])
    df_final |> CSV.write(joinpath(pwd(), "data", "twitter_timeseries.csv"))
    return nothing
end

# TODO: process_srag
