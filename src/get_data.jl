using Downloads: download

function download_srag(; output_file="srag_raw")
    output_file = joinpath(pwd(), "data", output_file)

    urls = [
        "2019" => "https://opendatasus.saude.gov.br/dataset/e99cfd21-3d8c-4ff9-bd9c-04b8b2518739/resource/9d1165b3-80a3-4ec4-a6ad-e980e3d354b2/download/influd19_limpo-27.04.2020-final.csv",
        "2020" => "https://s3-sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2020/INFLUD-02-08-2021.csv",
        "2021" => "https://s3-sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2021/INFLUD21-02-08-2021.csv"]

    Threads.@threads for url ∈ urls
        filename = output_file * "_$(url.first).csv"
        !isfile(filename) ? download(url.second, filename) : nothing
    end
    return nothing
end

function download_twitter(; output_file="twitter_raw")
    output_file = joinpath(pwd(), "data", output_file)

    urls = [
        "2019" => "https://zenodo.org/record/5073680/files/Brazil_Portuguese_COVID19_Tweets2019.csv?download=1",
        "2020" => "https://zenodo.org/record/5073680/files/Brazil_Portuguese_COVID19_Tweets2020.csv?download=1",
        "2021" => "https://zenodo.org/record/5073680/files/Brazil_Portuguese_COVID19_Tweets2021.csv?download=1"]

    Threads.@threads for url ∈ urls
        filename = output_file * "_$(url.first).csv"
        !isfile(filename) ? download(url.second, filename) : nothing
    end
    return nothing
end

download_srag()
download_twitter()
