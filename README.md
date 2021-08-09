# Classificador se Sintomas do COVID-19

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

Este projeto é um projeto de pesquisa do [LabCidades](https://github.com/LabCidades/) da [Universidade Nove de Julho -- UNINOVE](https://uninove.br).

## Dados
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5073680.svg)](https://doi.org/10.5281/zenodo.5073680)

Os dados do twitter podem ser encontrados em [10.5281/zenodo.5073680](https://doi.org/10.5281/zenodo.5073680).

As bases SRAG do SUS podem ser acessadas em:
* [SRAG 2019](https://opendatasus.saude.gov.br/dataset/bd-srag-2019)
* [SRAG 2020](https://opendatasus.saude.gov.br/dataset/bd-srag-2020)
* [SRAG 2021](https://opendatasus.saude.gov.br/dataset/bd-srag-2021)


As funções `download_twitter` e `download_srag` no arquivo `src/get_data.jl` fazem o download automático dos dados necessários para a reprodução das análises e código desse repositório.

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
`evo` | alta

## Equipe

* Pesquisador Responsável: [Jose Storopoli](https://github.com/storopoli)
* Pesquisador Associado: [Alessandra Pellini](https://github.com/acgpellini)
* Pesquisador Assistente: [Andre Santos](https://github.com/andrelmfsantos)
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

