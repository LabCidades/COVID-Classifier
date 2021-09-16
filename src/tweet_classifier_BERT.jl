using CSV
using CUDA
using CategoricalArrays
using DataFrames
using Flux
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using Transformers.BidirectionalEncoder
#using Transformers.HuggingFace
using WordTokenizers
using Flux.Data: DataLoader
using Flux: gradient
import Flux.Optimise: update!
using Random: seed!, shuffle
using BSON: @save

# taken from: https://github.com/chengchingwen/Transformers.jl/blob/master/example/BERT/clf.jl

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
enable_gpu(true)
seed!(123)
const regex_twitter = r"(@[A-Za-z0-9]+)"

# Load Data
file = joinpath(pwd(), "annotated_tweets", "2021_09_06_9k.csv")
df = DataFrame(
    CSV.File(
        file;
        types=Dict("id" => String),
        drop=[
            "video",
            "user_id",
            "conversation_id",
            "thumbnail",
            "near",
            "geo",
            "user_rt_id",
            "user_rt",
            "retweet_id",
            "retweet_date",
        ],
        missingstrings=["", "???"],
    ),
)

# Data Cleaning
dropmissing!(df, :label)
select!(df, :id, :date, :tweet, :label)
# Remove twitter handles
transform!(df, :tweet => ByRow(x -> replace(x, regex_twitter => "")); renamecols=false)
# label
# 1 and 3 is signal
# 0, 2, 4, 5 and 6 is noise
function replace_labels(label::Int64)
    return label == 1 || label == 3 ? 1 : 0
end
transform!(df, :label => ByRow(replace_labels); renamecols=false)

# Train/Test Split
function partitionTrainTest(data; at=0.9)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at * n))
    test_idx = view(idx, (floor(Int, at * n) + 1):n)
    return data[train_idx, :], data[test_idx, :]
end
train_df, test_df = partitionTrainTest(df; at=0.9)

# # Train/Test DataLoader
train_loader = DataLoader(
    (train_df[:, :tweet], train_df[:, :label]); batchsize=32, shuffle=true
)
test_loader = DataLoader(
    (test_df[:, :tweet], test_df[:, :label]); batchsize=32, shuffle=true
)

# Load Bert
# Doesn't work on BERT-imbau :(
# _bert_model, wordpiece, tokenizer = hgf"neuralmind/bert-large-portuguese-cased:model"
# This will download a huge file in your /home/storopoli/.julia/datadeps
_bert_model, wordpiece, tokenizer = pretrain"Bert-multilingual_L-12_H-768_A-12"
const vocab = Vocabulary(wordpiece)
const hidden_size = size(_bert_model.classifier.pooler.W, 1)
const clf = gpu(Chain(
    Dropout(0.1),
    Dense(hidden_size, 1), # binary classification
    logsoftmax,
))
const bert_model = gpu(
    set_classifier(_bert_model, (pooler=_bert_model.classifier.pooler, clf=clf))
)

const ps = params(bert_model)
const opt = ADAM(3.5e-5)
function loss(data, label, batchsize; mask=nothing)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)
    p = bert_model.classifier.clf(bert_model.classifier.pooler(t[:, 1, :]))
    p = reshape(p, batchsize)
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
    return (tok=tok, segment=segment), label, mask
end

# Accuracy
function acc(p, label)
    return sum(p .== label) / length(label)
end

# Train
function train!(epoch, train_loader, test_loader)
    @info "start training"
    for e in 1:epoch
        @info "epoch: $e"
        i = 1
        al::Float32 = 0.0
        for batch in train_loader
            data, label, mask = todevice(preprocess(batch[1], batch[2]))
            (l, p), back = Flux.pullback(ps) do
                loss(data, label, train_loader.batchsize; mask=mask)
            end
            #@show l
            a = acc(p, label)
            al += a
            grad = back((Flux.Zygote.sensitivity(l), nothing))
            i += 1
            update!(opt, ps, grad)
            #@show al / i
        end
        test()
    end
end

# Test
function test(test_loader)
    Flux.testmode!(bert_model)
    i = 1
    al::Float32 = 0.0
    for batch in test_loader
        data, label, mask = todevice(preprocess(batch[1], batch[2]))
        l, p = loss(data, label, test_loader.batchsize; mask=mask)
        #@show l
        a = acc(p, label)
        al += a
        i += 1
    end
    al /= i
    Flux.testmode!(bert_model, false)
    @show al
end

train!(2, train_loader, test_loader)

weights = params(bert_model);

@save joinpath(pwd(), "model_weights", "bert_model_ADAM_3.5e-5.bson") weights

