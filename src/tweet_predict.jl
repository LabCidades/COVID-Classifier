using ArgParse
using CSV
using CUDA
using DataFrames
using Dates
using Flux
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using Transformers.BidirectionalEncoder
using WordTokenizers
using Zygote
using BSON: @load
using Dates: now, canonicalize

# ArgParse
function parse_commandline()
    s = ArgParseSettings(;
        description="This script uses a pre-trained BERT transformer model to make predictions for signal (1) or noise (0) for the presence of COVID-19 symptoms in tweets.",
        autofix_names=true,
    )
    @add_arg_table! s begin
        "model-weights"
        help = "file of the model saved weights"
        arg_type = String
        required = true
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    enable_gpu(false)

    # Get Model
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
    # Load Bert
    # This will download a huge file in your ~/.julia/datadeps
    # Doesn't work on BERTimbau :(
    #_bert_model, wordpiece, tokenizer = hgf"neuralmind/bert-large-portuguese-cased:model"
    _bert_model, wordpiece, tokenizer = pretrain"Bert-multilingual_L-12_H-768_A-12"
    vocab = Vocabulary(wordpiece)
    hidden_size = size(_bert_model.classifier.pooler.W, 1)
    clf = cpu(Chain(
        # Dropout(0.1), # maybe train without Dropout?
        Dense(hidden_size, 1), # binary classification
        logsoftmax,
    ))
    bert_model = cpu(
        set_classifier(_bert_model, (pooler=_bert_model.classifier.pooler, clf=clf))
    )
    @load joinpath(pwd(), "model_weights", "$(parsed_args["model_weights"]).bson") weights
    Flux.loadparams!(bert_model, weights)
    Flux.testmode!(bert_model)
    t0 = now()
    for year in [2019, 2020, 2021]
        t1 = now()
        @info "Predicting in CPU year $(year)"
        df = load_data(year)
        df.label = get_predictions(df; bert_model, wordpiece, tokenizer, vocab)
        df |> CSV.write(
            joinpath(
                pwd(),
                "predictions",
                "$(parsed_args["model_weights"])-twitter_pred_$(year).csv",
            ),
        )
        @info "Prediction done. Elapsed: $(canonicalize(now() - t1))"
    end
    @info "All predictions done. Elased Total: $(canonicalize(now() - t0))"
end

function load_data(year::Int)
    regex_twitter = r"(@[A-Za-z0-9]+)"
    df =
        CSV.File(
            joinpath(pwd(), "data", "twitter_raw_$(year).csv");
            types=Dict(:id => String, :date => DateTime, :tweet => String),
            select=[:id, :date, :tweet],
            dateformat="yyyy-mm-dd HH:MM:SS",
        ) |> DataFrame
    transform!(df, :tweet => ByRow(x -> replace(x, regex_twitter => "")); renamecols=false)
    return df
end;

# Mask
markline(sent) = ["[CLS]"; sent; "[SEP]"];

# Preprocess Data
function preprocess(batch; wordpiece, tokenizer, vocab)
    sentence = markline.(wordpiece.(tokenizer.(batch)))
    mask = getmask(sentence)
    tok = vocab(sentence)
    segment = fill!(similar(tok), 1)
    return (tok=tok, segment=segment), mask
end;

function get_predictions(df::DataFrame; bert_model, wordpiece, tokenizer, vocab)
    data, mask = todevice(preprocess(df[:, :tweet]; wordpiece, tokenizer, vocab))
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)
    p = bert_model.classifier.clf(bert_model.classifier.pooler(t[:, 1, :]))
    p = Int.(reshape(p, nrow(df)))
    return p
end;

main()

