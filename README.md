# Classificador de Sintomas do COVID-19

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

Este projeto é um projeto de pesquisa do [LabCidades](https://github.com/LabCidades/) da [Universidade Nove de Julho - UNINOVE](https://uninove.br).

## Dados
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5073680.svg)](https://doi.org/10.5281/zenodo.5073680)

Os dados do twitter podem ser encontrados em [10.5281/zenodo.5073680](https://doi.org/10.5281/zenodo.5073680).

As bases SRAG do SUS podem ser acessadas em:
* [SRAG 2019](https://opendatasus.saude.gov.br/dataset/bd-srag-2019)
* [SRAG 2020](https://opendatasus.saude.gov.br/dataset/bd-srag-2020)
* [SRAG 2021](https://opendatasus.saude.gov.br/dataset/bd-srag-2021)

## Reprodução do Ambiente de Desenvolvimento

1. Clone o repositório e acesse o diretório raiz

   ```bash
   git clone https://github.com/LabCidades/COVID-Classifier.git
   cd COVID-Classifier
   ```
### Preparação dos Dados (Julia)

As funções `download_twitter` e `download_srag` no arquivo `src/get_data.jl` fazem o download automático dos dados; e as funções `process_twitter` e `process_srag` no arquivo `src/process_data.jl` fazem todo o processamento e manipulação de dados necessária para a reprodução das análises e código desse repositório.

Para reproduzir o ambiente de preparação dos dados:

1. [Instale Julia](https://julialang.org/downloads/)

2. Instancie o ambiente julia abrindo uma nova sessão de julia e digitando:

   ```julia
   using Pkg

   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. Em um terminal digite:
   1. `julia --project get_data.jl` para fazer o download dos arquivos de dados do SUS e do Zenodo (OBS: isto baixará cerca de 2.6GB de dados)
   2. `julia --project process_data.jl` para processar os arquivos de dados

### Classificador dos Tweets (PyTorch - Python)

Para reproduzir o ambiente de preparação dos dados:

1. [Instale Python](https://python.org) (recomendamos o [miniforge](https://github.com/conda-forge/miniforge) que é o anaconda opensource)

2. Instancie o ambiente Python digitando no terminal com o `conda` instalado:

   ```bash
    conda create -f environment.yml
   ```

3. O script `tweet_classifier_BERT.py` treina um classificador de tweets em sinal (1) ou ruído (0) para a presença de sintomas usando um modelo transformer BERT pré-treinado em português [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased):

   ```bash
   $ python src/tweet_classifier_BERT.py -h

   There are 1 GPU(s) available.
   We will use the GPU: GeForce RTX 3070 Ti
   usage: tweet_classifier_BERT.py [-h] [-f FILE]  [-lr LEARNING_RATE] [-e EPOCH] [-b BATCHSIZE]

   Este script treina um classificador de tweets em sinal (1) ou ruído (0) para a presença de sintomas usando um modelo transformer BERT pré-treinado em português BERTimbau

   optional arguments:
     -h, --help            show this help message and exit
     -f FILE, --file FILE  arquivo de treino com tweets rotulados
     -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                           taxa de aprendizagem, do paper do BERT você pode escolher dentre 5e-5, 3e-5 ou 2e-5
     -e EPOCH, --epoch EPOCH
                           épocas, do paper do BERT você pode escolher entre 2 a 4
     -b BATCHSIZE, --batchsize BATCHSIZE
                        tamanho do batch, ideal ser uma potência de 2, escolha com cuidado para não estourar a memória da GPU

   ```

   Os modelos pré-treinados do huggingface serão salvos no diretório `huggingface_cache/`

   Os pesos do treino serão salvos no diretório `model_weights/` com a seguinte assinatura de arquivo `{file}-lr_{learning_rate}-e_{epoch}-batch{batchsize}.pt`

   Os resultados do treino serão salvos no diretório `results/` com a seguinte assinatura de arquivo `{file}-lr_{learning_rate}-e_{epoch}-batch{batchsize}.csv`

4. O script `tweet_predict.py` usa o modelo treinado na etapa anterior e classifica os tweets em sinal (1) ou ruído (0) para a presença de sintomas:

   ```bash
   python src/tweet_predict.py -h
   There are 1 GPU(s) available.
   We will use the GPU: GeForce RTX 3070 Ti
   usage: tweet_predict.py [-h] [-f FILE] [-lr LEARNING_RATE] [-e EPOCH] [-b BATCHSIZE]

   Este script usa o modelo treinado na etapa anterior e classifica os tweets em sinal (1) ou ruído (0) para a presença de sintomas

   optional arguments:
     -h, --help            show this help message and exit
     -f FILE, --file FILE  arquivo de treino com tweets rotulados
     -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                           taxa de aprendizagem do modelo que você deseja usar
     -e EPOCH, --epoch EPOCH
                           épocas do modelo que você deseja usar
     -b BATCHSIZE, --batchsize BATCHSIZE
                           tamanho do batch do modelo que você deseja usar
   ```

   Assim como na etapa anterior, os modelos pré-treinados do huggingface serão carregados do diretório `huggingface_cache/`

   Os pesos do treino serão carregados do diretório `model_weights/` com a seguinte assinatura de arquivo `{file}-lr_{learning_rate}-e_{epoch}-batch{batchsize}.pt`

   As predições serão salvas no diretório `predictions/` com a seguinte assinatura de arquivo `twitter_pred_{year}.csv` com três colunas: `id`, `tweet` e `label`.

## Código dos Sintomas

Código | Sintoma
--- | ---
s01 | adinamia
s02 | ageusia
s03 | anosmia
s04 | boca azulada
s05 | calafrio
s06 | cansaço
s07 | cefaleia
s08 | cianose
s09 | coloração azulada no rosto
s10 | congestão nasal
s11 | conjuntivite
s12 | coriza
s13 | desconforto respiratório
s14 | diarreia
s15 | dificuldade para respirar
s16 | diminuição do apetite
s17 | dispneia
s18 | distúrbio gustativo
s19 | distúrbio olfativo
s20 | dor abdominal
s21 | dor de cabeça
s22 | dor de garganta
s23 | dor no corpo
s24 | dor no peito
s25 | dor persistente no tórax
s26 | erupção cutânea na pele
s27 | fadiga
s28 | falta de ar
s29 | febre
s30 | gripe
s31 | hiporexia
s32 | inapetência
s33 | infecção respiratória
s34 | lábio azulado
s35 | mialgia
s36 | nariz entupido
s37 | náusea
s38 | obstrução nasal
s39 | perda de apetite
s40 | perda do olfato
s41 | perda do paladar
s42 | pneumonia
s43 | pressão no peito
s44 | pressão no tórax
s45 | prostração
s46 | quadro gripal
s47 | quadro respiratório
s48 | queda da saturação
s49 | resfriado
s50 | rosto azulado
s51 | saturação baixa
s52 | saturação de o2 menor que 95%
s53 | síndrome respiratória aguda grave
s54 | SRAG
s55 | tosse
s56 | vômito

## Códigos SRAG `hospital`

Código | Descrição
--- | ---
`pri` | primeiros sintomas
`int` | hospitalização por SRAG
`ent` | entrada na UTI
`sai` | saída da UTI
`evo` | alta ou óbito

## Equipe

* Pesquisador Responsável: [Jose Storopoli](https://github.com/storopoli)
* Pesquisador Associado: [Alessandra Pellini](https://github.com/acgpellini)
* Pesquisador Assistente: [André Santos](https://github.com/andrelmfsantos)
* Alunos de Iniciação Científica:
  * [João Vinícius Vieira Nóia](https://github.com/vinivieiran)
  * [Elias Noda](https://github.com/Elias-Noda)
  * [Paula Fraga](https://github.com/Paula-Fraga)
  * [Camila Brichta](https://github.com/camibrichta)
  * [Leandro dos Santos](https://github.com/leandrors91)
  * [Junior De Sousa Silva](https://github.com/juniorghostinthewires)

## Licença

Esta obra está licenciada com uma Licença [Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0 Internacional][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

