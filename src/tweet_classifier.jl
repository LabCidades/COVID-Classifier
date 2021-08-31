using CSV
using CategoricalArrays
using DataFrames
using Languages
using MLJ
using TextAnalysis
using MLJ: fit!
using Unicode: normalize

const regex_twitter = r"(@[A-Za-z0-9]+)"

# Load Data
file = joinpath(pwd(), "annotated_tweets", "2021_08_31_8k.csv")
df = CSV.File(
    file,
    types=Dict(
        "id" => String,
        "conversation_id" => String,
        "user_id" => String,
        "video" => Bool),
    drop=["thumbnail", "near", "geo", "user_rt_id", "user_rt", "retweet_id", "retweet_date"],
    missingstrings = ["", "???"]) |>
        DataFrame

# Data Cleaning
dropmissing!(df, :label)
DataFrames.transform!(df, [:Column1, :hashtags, :urls, :photos, :reply_to, :symptoms] .=> categorical; renamecols=false)
# Remove twitter handles
DataFrames.transform!(df, :tweet => ByRow(x -> replace(x, regex_twitter => "")); renamecols=false)
# Remove Unicode Stuff for StringDocument
DataFrames.transform!(df, :tweet => ByRow(x -> remove_corrupt_utf8(normalize(x, casefold=true, stripcc=true, stripmark=true, stable=true, compat=true))); renamecols=false)

# label
# 1 and 3 is signal
# 0, 2, 4, 5 and 6 is noise
function replace_labels(label::Int64)
    return label == 1 || label == 3 ? 1 : 0
end
DataFrames.transform!(df, :label => ByRow(replace_labels); renamecols=false)

docs = StringDocument.(df[:, :tweet])
map(doc -> language!(doc, Languages.Portuguese()), docs)
remove_corrupt_utf8!.(docs)
remove_case!.(docs)
prepare!.(docs, strip_numbers | strip_punctuation | strip_stopwords)

crps = Corpus(docs)
update_lexicon!(crps)
m = DocumentTermMatrix(crps)
# X = dtm(m) # sparse matrix representation
X = tf_idf(m)
y = df[:, :label]

# Create a MLJ Model using EvoTree
