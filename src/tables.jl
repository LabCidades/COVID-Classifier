using PrettyTables

include(joinpath(pwd(), "src", "read_data.jl"))

# Twitter
# Sintomas totais
total_tweets = combine(groupby(tweets, [:symptom, :symptom_detail]), :n => sum)
sort!(total_tweets, :n_sum; rev=true)
rename!(total_tweets,
        :symptom => :ID,
        :symptom_detail => :Sintomas,
        :n_sum => :Tweets)
top_15_tweets = total_tweets[1:15, :]

formatter_twitter = (v, i, j) -> j == 3 ? replace(string(round(v / 1_000_000; digits=2)) * " mi", "." => ",") : v

pretty_table(top_15_tweets;
             nosubheader=true,
             title="Top-15 sintomas mais tuítados",
             formatters=formatter_twitter)

# SRAG
# Sintomas totais
srag_wide = unstack(srag, :hospital, :srag)
total_srag = combine(groupby(srag_wide, [:symptom, :symptom_detail]), [:ent, :sai, :evo] .=> sum)
sort!(total_srag, :ent_sum; rev=true)
rename!(total_srag,
        "symptom" => "ID",
        "symptom_detail" => "Sintomas",
        "ent_sum" => "Entrada na UTI",
        "sai_sum" => "Saída da UTI",
        "evo_sum" => "Alta")
top_30_srag = total_srag[1:30, :]

formatter_srag_million = (v, i, j) -> j ≥ 3 ? replace(string(round(v / 1_000_000; digits=2)) * " mi", "." => ",") : v
formatter_srag_thousands = (v, i, j) -> j ≥ 3 ? replace(string(round(v / 1_000; digits=1)) * " mil", "." => ",") : v

pretty_table(top_30_srag;
             nosubheader=true,
             title="Top-30 sintomas SRAG",
             formatters=formatter_srag_thousands)

top_20_all = leftjoin(total_srag[1:20, :], total_tweets, on = [:ID, :Sintomas])
pretty_table(top_20_all;
             nosubheader=true,
             title="Top-20 sintomas SRAG com Twitter",
             formatters=formatter_srag_million)
