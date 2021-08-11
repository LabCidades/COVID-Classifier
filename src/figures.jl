using AlgebraOfGraphics
using CairoMakie
using Distributed: pmap

include(joinpath(pwd(), "src", "read_data.jl"))

function minmax_scaler(x::AbstractVector; min=min(x...), max=max(x...))
    return (x .- min) ./ (max - min)
end

function range_scaler(x::AbstractVector; a=0, b=1)
    x = minmax_scaler(x)
    return (b - a) .*  x .+ a
end

# you need AlgebraOfGraphics version 0.5.1 for this
# Sintomas SRAG vs Twitter
function srag_vs_twitter(symptom::Symbol, hospital::String; df_tweets=tweets, df_srag=srag, vaccination=true, span=0.75, degree=2)
    symptoms_dict = Dict(
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
    hospital_dict = Dict(
        "pri" => "Primeiros Sintomas",
        "int" => "Hospitalização por SRAG",
        "ent" => "Entrada na UTI",
        "sai" => "Saída da UTI",
        "evo" => "Alta"
    )
    df = leftjoin(df_srag, df_tweets; on=[:date, :date_rata, :year, :month, :fortnight, :week, :symptom, :symptom_detail])
    symptom_df = stack(
        filter(row -> row.symptom == string(symptom) && row.hospital == hospital, df),
        [:srag, :n])
    months = min(symptom_df.date...):Month(1):max(symptom_df.date...)
    # pass formatting function and list of date ticks (AoG v 0.5.1)
    dateticks = AlgebraOfGraphics.datetimeticks(x -> Dates.format(x, dateformat"mm-yy"), months)
    transform!(symptom_df, :date_rata => x -> range_scaler(x; a=min(dateticks[1]...), b=max(dateticks[1]...)); renamecols=false)
    # Vacinação inicío em 17/01/2021
    janeiro = findfirst(x -> x == "01-21", dateticks[2])
    fevereiro = findfirst(x -> x == "02-21", dateticks[2])
    mid_jan_fev = (dateticks[1][janeiro] + dateticks[1][fevereiro]) / 2
    # Gráfico
    resolution = (800, 600)
    f = Figure(; resolution)
    plt = data(symptom_df) *
            mapping(:date_rata=>"Data",
                    :value => "",
                    color=:variable => renamer("srag" => hospital_dict[hospital], "n" => "Tweets com Sintoma") => "Cor")
    plt *= smooth(span=span, degree=degree)
    ag = draw!(f, plt;
               axis=(;
               title = "$(hospital_dict[hospital]) - $(titlecase(symptoms_dict[string(symptom)])) - ($(string(symptom)))",
               titlesize=20,
               xticklabelrotation=π/4,
               xlabelpadding=4.0,
               xticks=dateticks,
               ytickformat=(x -> string.(x ./ 1_000) .* "K")
           ))
    if vaccination
        vlines!(f.content[1], mid_jan_fev; color=:red ,linewidth=2) # vacinação
    end
    legend!(f[end+1, 1], ag; orientation=:horizontal, tellheight=true, tellwidth=false)
    return f
end

function save_figure(fig::Figure, filename::String, prefix::String; quality=3)
    save(joinpath(pwd(), "figures", "$(prefix)_$(filename).png"), fig, px_per_unit=quality)
end

# Top 5 Tweets
# symptoms_vec = [:s21, :s06, :s29, :s30, :s11, :s28, :s56, :s23, :s22, :s55, :s24, :s49, :s14, :s36, :s42]
symptoms_vec = [:s21, :s06, :s29, :s30, :s11, :s28]
hospital_vec = ["pri", "int", "ent", "sai", "evo"]
symptoms_hospital_vec = Iterators.product(symptoms_vec, hospital_vec) |> collect |> vec
figures_vec = map(x -> srag_vs_twitter(first(x), last(x); span=0.05, degree=20), symptoms_hospital_vec)
map(
    (fig, symptom_hospital) -> save_figure(
                                fig,
                                string(first(symptom_hospital)) * "_" * string(last(symptom_hospital)),
                                "srag_twitter", quality=3),
    figures_vec, symptoms_hospital_vec)
