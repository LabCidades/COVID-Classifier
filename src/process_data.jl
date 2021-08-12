using CSV
using Dates
using DataFrames
using Unicode

include(joinpath(pwd(), "src", "get_data.jl"))

# SRAG
# function process_srag()
    srag2019 = CSV.read(joinpath(pwd(), "data", "srag_raw_2019.csv"), DataFrame)
    srag2020 = CSV.read(joinpath(pwd(), "data", "srag_raw_2020.csv"), DataFrame)
    srag2021 = CSV.read(joinpath(pwd(), "data", "srag_raw_2021.csv"), DataFrame)

    # intersect significa concatenar apenas as colunas em comuns
    srag = vcat(srag2019, srag2020, srag2021; cols=:intersect)

    # Output tem que ser
    # data | hospital | srag
    # DT_NOTIFIC (data notificação)
    # DT_SIN_PRI (data primeiros sintomas)
    # DT_INTERNA (data internação por SRAG)
    # DT_ENTUTI (data entrada na UTI)
    # DT_SAIDUTI (data de saída da UTI)
    # DT_EVOLUCA (data de óbito ou evolução)

    select!(srag, :DT_NOTIFIC,
                  :DT_SIN_PRI => :pri,
                  :DT_INTERNA => :int,
                  :DT_ENTUTI => :ent,
                  :DT_SAIDUTI => :sai,
                  :DT_EVOLUCA => :evo,
                  :FEBRE, :TOSSE, :GARGANTA, :DISPNEIA, :DESC_RESP, :SATURACAO, :DIARREIA, :VOMITO, :OUTRO_DES)
    srag_pri = select(filter(row -> !ismissing(row.pri), srag), :pri, Between(:FEBRE, :OUTRO_DES))
    srag_int = select(filter(row -> !ismissing(row.int), srag), :int, Between(:FEBRE, :OUTRO_DES))
    srag_ent = select(filter(row -> !ismissing(row.ent), srag), :ent, Between(:FEBRE, :OUTRO_DES))
    srag_sai = select(filter(row -> !ismissing(row.sai), srag), :sai, Between(:FEBRE, :OUTRO_DES))
    srag_evo = select(filter(row -> !ismissing(row.evo), srag), :evo, Between(:FEBRE, :OUTRO_DES))

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

    for df ∈ [srag_pri, srag_int, srag_ent, srag_sai, srag_evo]
        for symptom ∈ symptoms_srag
            transform!(df, symptom.first => ByRow(x -> coalesce(x, 9) == 1) => symptom.second)
        end
    end

    for (k, v) ∈ missing_symptoms_dict
        missing_symptoms_dict[k] = string(uppercase(Unicode.normalize(v; casefold=true, stripmark=true)))
    end

    for df ∈ [srag_pri, srag_int, srag_ent, srag_sai, srag_evo]
        for (k, v) ∈ missing_symptoms_dict
            transform!(df, :OUTRO_DES => ByRow(x -> occursin(v, coalesce(x, ""))) => k)
        end
    end

    # faz um df longo com os sintomas e ja filtrando com os valores que aparecem apenas
    symptoms_cols = names(srag_pri, r"^s\d{2}")
    srag_pri_long = filter(row -> row.value_to_keep == true, stack(srag_pri, symptoms_cols, variable_name=:symptom, value_name=:value_to_keep))
    srag_int_long = filter(row -> row.value_to_keep == true, stack(srag_int, symptoms_cols, variable_name=:symptom, value_name=:value_to_keep))
    srag_ent_long = filter(row -> row.value_to_keep == true, stack(srag_ent, symptoms_cols, variable_name=:symptom, value_name=:value_to_keep))
    srag_sai_long = filter(row -> row.value_to_keep == true, stack(srag_sai, symptoms_cols, variable_name=:symptom, value_name=:value_to_keep))
    srag_evo_long = filter(row -> row.value_to_keep == true, stack(srag_evo, symptoms_cols, variable_name=:symptom, value_name=:value_to_keep))

    # agrupa os srag por data e tipo de SRAG
    pri_final = combine(groupby(srag_pri_long, [:pri, :symptom]), :pri, :symptom, nrow => :srag)
    int_final = combine(groupby(srag_int_long, [:int, :symptom]), :int, :symptom, nrow => :srag)
    ent_final = combine(groupby(srag_ent_long, [:ent, :symptom]), :ent, :symptom, nrow => :srag)
    sai_final = combine(groupby(srag_sai_long, [:sai, :symptom]), :sai, :symptom, nrow => :srag)
    evo_final = combine(groupby(srag_evo_long, [:evo, :symptom]), :evo, :symptom, nrow => :srag)
    transform!(pri_final, :pri => ByRow(x -> Date(x, dateformat"d/m/Y")) => :date)
    transform!(int_final, :int => ByRow(x -> Date(x, dateformat"d/m/Y")) => :date)
    transform!(ent_final, :ent => ByRow(x -> Date(x, dateformat"d/m/Y")) => :date)
    transform!(sai_final, :sai => ByRow(x -> Date(x, dateformat"d/m/Y")) => :date)
    transform!(evo_final, :evo => ByRow(x -> Date(x, dateformat"d/m/Y")) => :date)
    df_pri_final = combine(groupby(pri_final, [:date, :symptom]), :srag => sum => :srag)
    df_int_final = combine(groupby(int_final, [:date, :symptom]), :srag => sum => :srag)
    df_ent_final = combine(groupby(ent_final, [:date, :symptom]), :srag => sum => :srag)
    df_sai_final = combine(groupby(sai_final, [:date, :symptom]), :srag => sum => :srag)
    df_evo_final = combine(groupby(evo_final, [:date, :symptom]), :srag => sum => :srag)

    # define o tipo de srag na coluna hospital de cada df de srag
    df_pri_final[!, :hospital] .= "pri"
    df_int_final[!, :hospital] .= "int"
    df_ent_final[!, :hospital] .= "ent"
    df_sai_final[!, :hospital] .= "sai"
    df_evo_final[!, :hospital] .= "evo"

    # juntar tudo
    df_count = vcat(df_pri_final, df_int_final, df_ent_final, df_sai_final, df_evo_final)
    # transform!(df_count, :date => ByRow(x -> Date(x, dateformat"d/m/Y")); renamecols=false)

    # cria um dataframe para todas as datas e sintomas
    df_dates = DataFrame(date=Date("2019-01-01"):Day(1):Date("2021-06-30"))
    df_hospitals = DataFrame(hospital=["pri", "int", "ent", "sai", "evo"])
    df_symptoms = DataFrame(symptom=unique(df_count.symptom))
    df_union = crossjoin(df_dates, df_hospitals, df_symptoms)
    # junta tudo
    df_final = leftjoin(df_union, df_count, on=[:date, :hospital, :symptom])
    df_final.srag = coalesce.(df_final.srag, 0)
    df_final |> CSV.write(joinpath(pwd(), "data", "srag_timeseries.csv"))
    return nothing
# end

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
