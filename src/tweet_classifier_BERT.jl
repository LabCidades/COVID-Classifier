using CSV
using CUDA
using CategoricalArrays
using DataFrames
using Flux
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using Transformers.BidirectionalEncoder
using Transformers.HuggingFace
using Flux.Data: DataLoader
using Flux: gradient
import Flux.Optimise: update!
using Random: seed!, shuffle
using BSON: @save

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
enable_gpu(true)
seed!(123)
const regex_twitter = r"(@[A-Za-z0-9]+)"

# Load Data
file = joinpath(pwd(), "annotated_tweets", "2021_08_31_8k.csv")
df =
    CSV.File(
        file,
        types = Dict(
            "id" => String,
            "conversation_id" => String,
            "user_id" => String,
            "video" => Bool,
        ),
        drop = [
            "thumbnail",
            "near",
            "geo",
            "user_rt_id",
            "user_rt",
            "retweet_id",
            "retweet_date",
        ],
        missingstrings = ["", "???"],
    ) |> DataFrame

# Data Cleaning
dropmissing!(df, :label)
transform!(
    df,
    [:Column1, :hashtags, :urls, :photos, :reply_to, :symptoms] .=> categorical;
    renamecols = false,
)
# Remove twitter handles
transform!(df, :tweet => ByRow(x -> replace(x, regex_twitter => "")); renamecols = false)
# label
# 1 and 3 is signal
# 0, 2, 4, 5 and 6 is noise
function replace_labels(label::Int64)
    return label == 1 || label == 3 ? 1 : 0
end
transform!(df, :label => ByRow(replace_labels); renamecols = false)

# Train/Test Split
function partitionTrainTest(data; at = 0.8)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at * n))
    test_idx = view(idx, (floor(Int, at * n) + 1):n)
    data[train_idx, :], data[test_idx, :]
end
train_df, test_df = partitionTrainTest(df; at = 0.8)

# # Train/Test DataLoader
train_loader =
    DataLoader((train_df[:, :tweet], train_df[:, :label]), batchsize = 32, shuffle = true)
test_loader =
    DataLoader((test_df[:, :tweet], test_df[:, :label]), batchsize = 32, shuffle = true)

# Load Bert
# Doesn't work on BERT-imbau
_bert_model, wordpiece, tokenizer = hgf"neuralmind/bert-large-portuguese-cased:model"
# This will download a huge file in your /home/storopoli/.julia/datadeps
_bert_model, wordpiece, tokenizer = pretrain"Bert-multilingual_L-12_H-768_A-12"
const mywordpiece = wordpiece(df[:, :tweet])
const vocab = Vocabulary(df[:, :tweet])
const hidden_size = size(_bert_model.classifier.pooler.W, 1)
const clf = gpu(Chain(
    Dropout(0.1),
    Dense(hidden_size, 1), # binary classification
    logsoftmax,
))
const bert_model =
    gpu(set_classifier(_bert_model, (pooler = _bert_model.classifier.pooler, clf = clf)))

const ps = params(bert_model)
const opt = ADAM(1e-6)
function loss(data, label, mask = nothing)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)

    p = bert_model.classifier.clf(bert_model.classifier.pooler(t[:, 1, :]))

    l = Flux.logitbinarycrossentropy(label, p)
    return l, p
end

markline(sent) = ["[CLS]"; sent; "[SEP]"]

# Preprocess Data
function preprocess(batch, label)
    sentence = markline.(wordpiece.(tokenizer.(batch)))
    mask = getmask(sentence)
    tok = vocab(sentence)
    segment = fill!(similar(tok), 1)
    return (tok = tok, segment = segment), label, mask
end

# Accuracy
function acc(p, label)
    sum(p .== label) / length(label)
end

# Train
function train!(epoch)
    @info "start training"
    for e = 1:epoch
        @info "epoch: $e"
        i = 1
        al::Float64 = 0.0
        for batch in train_loader
            data, label, mask = todevice(preprocess(batch[1], batch[2]))
            l, p = loss(data, label, mask)
            @show l
            a = acc(p, label)
            al += a
            grad = gradient(() -> l, ps)
            i += 1
            update!(opt, ps, grad)
            @show al / i
        end
        test()
    end
end

# Test
function test()
    Flux.testmode!(bert_model)
    i = 1
    al::Float64 = 0.0
    for batch in test_loader
        data, label, mask = todevice(preprocess(batch[1], batch[2]))
        l, p = loss(data, label, mask)
        @show l
        a = acc(p, label)
        al += a
        i += 1
    end
    al /= i
    Flux.testmode!(bert_model, false)
    @show al
end
