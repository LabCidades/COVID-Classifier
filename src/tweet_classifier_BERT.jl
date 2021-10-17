using ArgParse
using CSV
using CUDA
using CategoricalArrays
using DataFrames
using Flux
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using Transformers.BidirectionalEncoder
using WordTokenizers
using Zygote
using Flux.Data: DataLoader
using Flux: gradient
import Flux.Optimise: update!
using BSON: @save
using Dates: now, canonicalize
using MLBase: confusmat
using Random: seed!, shuffle

# taken from: https://github.com/chengchingwen/Transformers.jl/blob/master/example/BERT/clf.jl

# ArgParse
function parse_commandline()
    s = ArgParseSettings(;
        description="This script trains a tweet classifier in signal (1) or noise (0) for the presence of symptoms using a pre-trained BERT transformer model.",
        autofix_names=true,
    )
    @add_arg_table! s begin
        "--batchsize", "-b"
        help = "batchsize"
        arg_type = Int
        default = 32
        "--epoch", "-e"
        help = "epoch, choose between 2 to 4"
        arg_type = Int
        default = 3
        "--learning-rate", "-l"
        help = "learning rate, choose between 5e-5, 3e-5 or 2e-5"
        arg_type = Float64
        default = 5e-5
        "--seed", "-s"
        help = "random seed"
        arg_type = Int
        default = 123
        "--gpu", "-g"
        help = "use gpu"
        action = :store_true
        "file"
        help = "file of annotated tweets"
        arg_type = String
        required = true
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    # Random seed
    seed!(parsed_args["seed"])
    # GPU
    enable_gpu(parsed_args["gpu"])

    # Load Data
    df = load_data(parsed_args["file"])
    train_df, test_df = partitionTrainTest(df; at=0.9)

    # Train/Test DataLoader
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
    # My GPU only tolerates 8. 16 or 32 it blows up!
    train_loader = DataLoader(
        (train_df[:, :tweet], train_df[:, :label]);
        batchsize=parsed_args["batchsize"],
        shuffle=true,
    )
    @info "Training on $(nrow(train_df)) samples"
    test_loader = DataLoader(
        (test_df[:, :tweet], test_df[:, :label]);
        batchsize=parsed_args["batchsize"],
        shuffle=true,
    )
    @info "Testing on $(nrow(test_df)) samples"

    # Get Model
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
    # Load Bert
    # This will download a huge file in your ~/.julia/datadeps
    # Doesn't work on BERTimbau :(
    #_bert_model, wordpiece, tokenizer = hgf"neuralmind/bert-large-portuguese-cased:model"
    _bert_model, wordpiece, tokenizer = pretrain"Bert-multilingual_L-12_H-768_A-12"
    vocab = Vocabulary(wordpiece)
    hidden_size = size(_bert_model.classifier.pooler.W, 1)
    if parsed_args["gpu"] == true
        @info "Running on the GPU"
        clf = gpu(Chain(
            # Dropout(0.1), # maybe train without Dropout?
            Dense(hidden_size, 1), # binary classification
            logsoftmax,
        ))
        bert_model = gpu(
            set_classifier(_bert_model, (pooler=_bert_model.classifier.pooler, clf=clf))
        )
    else
        @info "Running on the CPU"
        clf = cpu(Chain(
            # Dropout(0.1), # maybe train without Dropout?
            Dense(hidden_size, 1), # binary classification
            logsoftmax,
        ))
        bert_model = cpu(
            set_classifier(_bert_model, (pooler=_bert_model.classifier.pooler, clf=clf))
        )
    end
    ps = params(bert_model)
    opt = ADAMW(parsed_args["learning_rate"], (0.9, 0.999), 0.01)
    train!(
        parsed_args["epoch"], train_loader; bert_model, opt, ps, wordpiece, tokenizer, vocab
    )
    results = test(test_loader; bert_model, wordpiece, tokenizer, vocab)
    DataFrame(; results...) |> CSV.write(
        joinpath(
            pwd(),
            "results",
            "$(parsed_args["file"])-lr_$(parsed_args["learning_rate"])-e_$(parsed_args["epoch"])-batch_$(parsed_args["batchsize"]).csv",
        ),
    )
    weights = params(bert_model)
    # Move model to CPU to save weights
    cpu(bert_model)
    @save joinpath(
        pwd(),
        "model_weights",
        "$(parsed_args["file"])-lr_$(parsed_args["learning_rate"])-e_$(parsed_args["epoch"])-batch_$(parsed_args["batchsize"]).bson",
    ) weights
end

# Load Data
function load_data(file::AbstractString)
    regex_twitter = r"(@[A-Za-z0-9]+)"
    df =
        CSV.File(
            joinpath(pwd(), "annotated_tweets", "$(file).csv");
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
            missingstring=["", "???"],
            limit=1_000,
        ) |> DataFrame
    # Data Cleaning
    dropmissing!(df, :label)
    select!(df, :id, :date, :tweet, :label)
    # Remove twitter handles
    transform!(df, :tweet => ByRow(x -> replace(x, regex_twitter => "")); renamecols=false)
    transform!(df, :label => ByRow(replace_labels); renamecols=false)
    return df
end;

# label
# 1 and 3 is signal
# 0, 2, 4, 5 and 6 is noise
function replace_labels(label::Int64)
    return label == 1 || label == 3 ? 1 : 0
end;

# Train/Test Split
function partitionTrainTest(data; at=0.9)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at * n))
    test_idx = view(idx, (floor(Int, at * n) + 1):n)
    return data[train_idx, :], data[test_idx, :]
end;

function loss(data, label, batchsize; mask=nothing, bert_model)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)
    p = bert_model.classifier.clf(bert_model.classifier.pooler(t[:, 1, :]))
    p = reshape(p, batchsize)
    l = Flux.logitbinarycrossentropy(label, p)
    return l, p
end;

# Mask
markline(sent) = ["[CLS]"; sent; "[SEP]"];

# Preprocess Data
function preprocess(batch, label; wordpiece, tokenizer, vocab)
    sentence = markline.(wordpiece.(tokenizer.(batch)))
    mask = getmask(sentence)
    tok = vocab(sentence)
    segment = fill!(similar(tok), 1)
    return (tok=tok, segment=segment), label, mask
end;

# Accuracy
function acc(p, label)
    return sum(p .== label) / length(label)
end;

# Train
function train!(epoch, train_loader; bert_model, opt, ps, wordpiece, tokenizer, vocab)
    t0 = now()
    @info "Start training"
    for e in 1:epoch
        @info "Epoch: $e / $(epoch)"
        a::Float32 = 0.0
        for (idx, batch) in enumerate(train_loader)
            # Progress update every 100 batches
            if idx % 100 == 0 && idx != 1
                @info "Batch: $(idx) / $(length(train_loader))\tElapsed: $(canonicalize(now() - t0))"
            end
            data, label, mask = todevice(
                preprocess(batch[1], batch[2]; wordpiece, tokenizer, vocab)
            )
            (l, p), back = Flux.pullback(ps) do
                loss(data, label, length(label); mask=mask, bert_model)
            end
            a += acc(p, label)
            grad = back((Flux.Zygote.sensitivity(l), nothing))
            update!(opt, ps, grad)
            # reclaim memory
            data = label = mask = nothing
            GC.gc(true)
            if idx == length(train_loader)
                @info "Epoch: $e\tLoss: $(round(l; digits=3))\tAccuracy: $(round(a / length(train_loader); digits=3))"
            end
        end
        # reclaim memory
        GC.gc(true)
    end
    # reclaim memory
    GC.gc(true)
    @info "Train Elapsed: $(canonicalize(now() - t0))"
end;

# Test
function test(test_loader; bert_model, wordpiece, tokenizer, vocab)
    t0 = now()
    Flux.testmode!(bert_model)
    conf_mat = zeros(Int, 2, 2)
    a::Float32 = 0.0
    for batch in test_loader
        data, label, mask = todevice(
            preprocess(batch[1], batch[2]; wordpiece, tokenizer, vocab)
        )
        _, p = loss(data, label, length(label); mask=mask, bert_model)
        a += acc(p, label)
        # The .+ 1 is because MLBase.jl throws an error if a class has label `0`.
        # So we shift everything by 1
        conf_mat += CUDA.@allowscalar confusmat(2, label .+ 1, Int.(p) .+ 1)
        # reclaim memory
        data = label = mask = nothing
        GC.gc(true)
    end
    a /= length(test_loader)
    tp = CUDA.@allowscalar conf_mat[1, 1] / sum(conf_mat[1, :])
    tn = CUDA.@allowscalar conf_mat[2, 2] / sum(conf_mat[2, :])
    fp = CUDA.@allowscalar conf_mat[1, 2] / sum(conf_mat[1, :])
    fn = CUDA.@allowscalar conf_mat[2, 1] / sum(conf_mat[2, :])
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * ((prec * rec) / (prec + rec))
    Flux.testmode!(bert_model, false)
    # reclaim memory
    GC.gc(true)
    @info "Test Accuracy: $(round(a; digits=3))"
    @info "Test Confusion Matrix:\n$(conf_mat)"
    @info "Test True Positive Rate: $(round(tp; digits=3))\t False Positive Rate: $(round(fp; digits=3))"
    @info "Test True Negative Rate: $(round(tn; digits=3))\t False Negative Rate: $(round(fn; digits=3))"
    @info "Test Precision: $(round(prec; digits=3))\tTest Recall: $(round(rec; digits=3))"
    @info "Test F1 Score: $(round(f1; digits=3))"
    @info "Test Elapsed: $(canonicalize(now() - t0))"
    return (; a, tp, fp, tn, fn, prec, rec, f1)
end;

main()
