using AlgebraOfGraphics
using CairoMakie

include(joinpath(pwd(), "src", "read_data.jl"))

# Pure Makie Implementation
# still need to figure out color palette
xs = min(tweets.date...):Day(1):max(tweets.date...)
months = min(tweets.date...):Month(1):max(tweets.date...)
lentime = length(xs)
slice_dates = 1:90:lentime
y_dor_de_cabeca = filter(row -> row.symptom == "s21", tweets).n
y_cansaco = filter(row -> row.symptom =="s06", tweets).n

# Gráfico
resolution = (800, 600)
f = Figure(; resolution)
ax = Axis(f[1,1];
          xlabel = "Data",
          ylabel = "Frequência em milhares")
l1 = lines!(ax, 1:length(xs), y_cansaco; color=:blue, label="cansaço")
l2 = lines!(ax, 1:length(xs), y_dor_de_cabeca; color=:red, label="dor de cabeca")
ax.xticks = (slice_dates, Dates.format.(xs, dateformat"mm-yy")[slice_dates])
ax.xticklabelrotation = π/4
ax.xlabelpadding = 4.0
axislegend("Sintoma"; position=:ct, orientation=:horizontal)
f

# AoG Implementation
# you need AlgebraOfGraphics#master for this

# Somente s21 e s06
n_21_06 = filter(row -> row.symptom == "s21" || row.symptom == "s06", tweets)
months = min(tweets.date...):Month(3):max(tweets.date...)
dateticks = AlgebraOfGraphics.datetimeticks(x -> Dates.format(x, dateformat"mm-yy"), months) # pass formatting function and list of date ticks
# Gráfico
resolution = (800, 600)
f = Figure(; resolution)
plt = data(n_21_06) *
        mapping(:date=>"Data",
                :n => "",
                color=:symptom => renamer("s21" => "dor de cabeça", "s06" => "cansaço") => "Sintomas") *
        visual(Lines)

ag = draw!(f, plt;
           axis=(;
           title = "Frequência de sintomas no Twitter",
           titlesize=20,
           xticklabelrotation=π/4,
           xlabelpadding=4.0,
           xticks=dateticks,
           ytickformat=(x -> string.(x ./ 1_000) .* "K")
           ))
legend!(f[end+1, 1], ag; orientation=:horizontal, tellheight=true, tellwidth=false)
f

# Sintomas SRAG vs Twitter
# TODO: Adicionar um vline! em 01/01/2021 - Vacinação
function srag_vs_twitter(symptom::Symbol, hospital::String; df_tweets=tweets, df_srag=srag)
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
    df = leftjoin(select(df_srag, Not(:tweet)), df_tweets; on=[:date, :symptom])
    symptom_df = stack(
        filter(row -> row.symptom == string(symptom) && row.hospital == hospital, df),
        [:srag, :n])
    transform!(symptom_df, :value => ByRow(x -> Int(x)); renamecols=false)
    months = min(symptom_df.date...):Month(3):max(symptom_df.date...)
    dateticks = AlgebraOfGraphics.datetimeticks(x -> Dates.format(x, dateformat"mm-yy"), months) # pass formatting function and list of date ticks
    # Gráfico
    resolution = (800, 600)
    f = Figure(; resolution)
    plt = data(symptom_df) *
            mapping(:date=>"Data",
                    :value => "",
                    color=:variable => renamer("srag" => hospital_dict[hospital], "n" => "Tweets com Sintoma") => "Cor") *
            visual(Lines)

    ag = draw!(f, plt;
               axis=(;
               title = "$(hospital_dict[hospital]) - $(titlecase(symptoms_dict[string(symptom)])) - ($(string(symptom)))",
               titlesize=20,
               xticklabelrotation=π/4,
               xlabelpadding=4.0,
               xticks=dateticks,
               ytickformat=(x -> string.(x ./ 1_000) .* "K")
           ))
    legend!(f[end+1, 1], ag; orientation=:horizontal, tellheight=true, tellwidth=false)
    return f
end
