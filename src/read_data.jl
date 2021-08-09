using CSV
using Dates
using DataFrames

include(joinpath(pwd(), "src", "get_data.jl"))

const symptoms_dict = Dict(
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
    "s13" => "desconforto respiratório",
    "s14" => "diarreia",
    "s15" => "dificuldade para respirar",
    "s16" => "diminuição do apetite",
    "s17" => "dispneia",
    "s18" => "distúrbio gustativo",
    "s19" => "distúrbio olfativo",
    "s20" => "dor abdominal",
    "s21" => "dor de cabeça",
    "s22" => "dor de garganta",
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
    "s52" => "saturação de o2 menor que 95%",
    "s53" => "síndrome respiratória aguda grave",
    "s54" => "SRAG",
    "s55" => "tosse",
    "s56" => "vômito"
)

function symptom_map(x::AbstractString, symptoms_dict::Dict{String, String})
    return symptoms_dict[x]
end

# no Twitter não temos s09 s18 s51
tweets = CSV.read(joinpath(pwd(), "data", "covid_twitter_time_series.csv"), DataFrame)
transform!(tweets, :n => ByRow(Int); renamecols=false)
transform!(tweets, :symptom => ByRow(x -> symptom_map(x, symptoms_dict)) => :symptom_detail)

# SRAG temos todos symptoms
srag = CSV.read(joinpath(pwd(), "data", "SRAG_time_series.csv"), DataFrame)
sort!(srag, [:date, :symptom])
transform!(srag, :tweet => ByRow(Int); renamecols=false)
unique(srag.hospital)

# Checagem básica
total_esperado = length(unique(df.symptom)) * length(Date("2019-01-01"):Day(1):Date("2021-06-30"))
total_tweets = combine(groupby(tweets, [:date, :symptom]), :n => sum => :n)
total_srag = combine(
    groupby(
        filter(row -> row.hospital == "ent", srag),
        [:date, :symptom]), :tweet => sum => :n)

# TODO: Agrupar por semana para suavizar os picos
