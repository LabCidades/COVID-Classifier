using CSV
using Dates
using DataFrames
using Unicode

include(joinpath(pwd(), "src", "get_data.jl"))

# SRAG

srag2019 = CSV.read(joinpath(pwd(), "data", "srag_raw_2019.csv"), DataFrame)
srag2020 = CSV.read(joinpath(pwd(), "data", "srag_raw_2020.csv"), DataFrame)
srag2021 = CSV.read(joinpath(pwd(), "data", "srag_raw_2021.csv"), DataFrame)

# intersect significa concatenar apenas as colunas em comuns
srag = vcat(srag2019, srag2020, srag2021; cols=:intersect)

# TODO: function process_srag()
# Output tem que ser
# data | hospital | srag
# DT_NOTIFIC (data notificação)
# DT_SIN_PRI (data primeiros sintomas)
# DT_INTERNA (data internação por SRAG)
# DT_ENTUTI (data entrada na UTI)
# DT_SAIDUTI (data de saída da UTI)
# DT_EVOLUCA (data de óbito ou evolução)

# Sintomas base SRAG (Vector de Pair)
symptoms_srag = [
    "FEBRE" => "s29",
    "TOSSE" => "s55",
    "GARGANTA" => "s22",
    "DISPNEIA" => "s17",
    "DESC_RESP" => "s13",
    "SATURACAO" => "s52",
    "DIARREIA" => "s14",
    "VOMITO" => "s56"
]

# Sintomas não presentes na base SRAG
missing_symptoms_dict = Dict(
    "s01" => "adinamia",
    "s02" => "ageusia",
    "s03" => "anosmia",
    "s04" => "boca azulada",
    "s05" => "calafrio",
    "s06" => "cansaço",
    "s07" => "cefaleia",
    "s08" => "cianose",
    "s09" => "coloração azulada no rosto",
    "s10" => "congestão nasal",
    "s11" => "conjuntivite",
    "s12" => "coriza",
    "s15" => "dificuldade para respirar",
    "s16" => "diminuição do apetite",
    "s18" => "distúrbio gustativo",
    "s19" => "distúrbio olfativo",
    "s20" => "dor abdominal",
    "s21" => "dor de cabeça",
    "s23" => "dor no corpo",
    "s24" => "dor no peito",
    "s25" => "dor persistente no tórax",
    "s26" => "erupção cutânea na pele",
    "s27" => "fadiga",
    "s28" => "falta de ar",
    "s29" => "febre",
    "s30" => "gripe",
    "s31" => "hiporexia",
    "s32" => "inapetência",
    "s33" => "infecção respiratória",
    "s34" => "lábio azulado",
    "s35" => "mialgia",
    "s36" => "nariz entupido",
    "s37" => "náusea",
    "s38" => "obstrução nasal",
    "s39" => "perda de apetite",
    "s40" => "perda do olfato",
    "s41" => "perda do paladar",
    "s42" => "pneumonia",
    "s43" => "pressão no peito",
    "s44" => "pressão no tórax",
    "s45" => "prostração",
    "s46" => "quadro gripal",
    "s47" => "quadro respiratório",
    "s48" => "queda da saturação",
    "s49" => "resfriado",
    "s50" => "rosto azulado",
    "s51" => "saturação baixa",
    "s53" => "síndrome respiratória aguda grave",
    "s54" => "SRAG",
)

srag.OUTRO_DES

sintomas_temp = join(unique(collect(skipmissing(srag.OUTRO_DES))));

for (k, v) ∈ missing_symptoms_dict
    println(Unicode.normalize(v))
end

for (k, v) ∈ missing_symptoms_dict
    if occursin(v, lowercase(sintomas_temp))
        println(k)
    end
end



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
    # cria um dataframe para todas as datas e sintomas
    df_dates = DataFrame(date=Date("2019-01-01"):Day(1):Date("2021-06-30"))
    df_symptoms = DataFrame(symptom=unique(tweets.symptoms))
    df_union = crossjoin(df_dates, df_symptoms)
    # combine os tweets df por data e sintoma e conta linhas como n
    df_count = combine(groupby(tweets, [:date, :symptoms]), nrow => :n)
    df_final = leftjoin(df_union, df_count, on=[:date, :symptom => :symptoms])
    transform!(df_final, :n => ByRow(x -> coalesce(x, 0)); renamecols=false)
    sort!(df_final, [:date, :symptom])
    df_final |> CSV.write(joinpath(pwd(), "data", "twitter_timeseries.csv"))
    return nothing
end
