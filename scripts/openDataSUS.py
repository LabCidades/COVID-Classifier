#!/usr/bin/env python
# coding: utf-8

# # Programa para ler o Banco de Dados de Síndrome Respiratória Aguda Grave (SRAG)

# __Base de dados__:
# * [SRAG2019](https://opendatasus.saude.gov.br/dataset/bd-srag-2019) ~ 022 KB
# * [SRAG2020](https://opendatasus.saude.gov.br/dataset/bd-srag-2020) ~ 727 KB
# * [SRAG2021](https://opendatasus.saude.gov.br/dataset/bd-srag-2021) ~ 638 KB

# In[1]:


# Carrega pacotes necessários
import pandas as pd
import numpy as np
import os


# In[2]:


# Carregar base de dados
os.chdir("C:/Users/andre/OneDrive/Pesquisas/ANDRE_TESE_COVID/TeseDados/SRAG")  # Aponta para a pasta que está o csv
srag = pd.read_csv("INFLUD020820212020.csv", sep = ";")                        # Carrega o csv
srag                                                                           # Exibi primeiras e últimas linhasda base


# In[3]:


# Dataframe com variáveis de interesse

# Variáveis:
# DT_NOTIFIC: Data da notificação
# DT_SIN_PRI: Data dos primeiros sintomas
# DT_INTERNA: Data da internação
# DT_ENTUTI:  Data da entrada na UTI
# DT_SAIDUTI: Data da saída da UTI
# DT_EVOLUCA: Data da alta
# FEBRE:      Paciente apresentou febre? 1-Sim; 2-Não; 9-Ignorado
# TOSSE:      Paciente apresentou tosse? 1-Sim; 2-Não; 9-Ignorado
# GARGANTA:   Paciente apresentou dor de garganta? 1-Sim; 2-Não; 9-Ignorado
# DISPNEIA:   Paciente apresentou dispneia? 1-Sim; 2-Não; 9-Ignorado
# DESC_RESP:  Paciente apresentou desconforto respiratório? 1-Sim; 2-Não; 9-Ignorado
# SATURAÇÃO:  Paciente apresentou saturação O2<95%? 1-Sim; 2-Não; 9-Ignorado
# DIARREIA:   Paciente apresentou diarreia? 1-Sim; 2-Não; 9-Ignorado
# VOMITO:     Paciente apresentou vômito? 1-Sim; 2-Não; 9-Ignorado
# DOR_ABD:    Paciente apresentou dor abdominal? 1-Sim; 2-Não; 9-Ignorado
# FADIGA:     Paciente apresentou fadiga? 1-Sim; 2-Não; 9-Ignorado
# PERD_OLFT:  Paciente apresentou perda de olfato? 1-Sim; 2-Não; 9-Ignorado
# PERD_PALA:  Paciente apresentou perda de paladar? 1-Sim; 2-Não; 9-Ignorado
# OUTRO_SIN:  Paciente apresentou outro(s) sintoma(s)? 1-Sim; 2-Não; 9-Ignorado

# Filtra casos de COVID-19
# CLASSI_FIN: 5-COVID-19; 1-Influenza; 2-Outro vírus; 3-Outro agente etiológico; 4-SRAG não especificado
df = srag[srag['CLASSI_FIN'] == 5]
df = df[['DT_NOTIFIC', 'DT_SIN_PRI', 'DT_INTERNA', 'DT_ENTUTI', 'DT_SAIDUTI', 'DT_EVOLUCA', 'FEBRE', 'TOSSE',
         'GARGANTA', 'DISPNEIA', 'DESC_RESP', 'SATURACAO', 'DIARREIA', 'VOMITO', 'DOR_ABD','FADIGA',
         'PERD_OLFT','PERD_PALA','OUTRO_SIN']]
df


# In[4]:


# Transpor colunas para linhas (Subset para fases de hospitalização)
dt = df.iloc[:,0:6]                             # cria dataframe com as colunas de datas (fases de hospitalização)
dt = dt.fillna(0)                               # substitui valores ausentes nas linhas por zero
dt = dt.set_index('DT_NOTIFIC')                 # coloca a coluna DT_NOTIFIC como índice
dt = dt.stack()                                 # transpor colunas em linhas
dt = pd.DataFrame(dt)                           # converte para formato dataframe
dt = dt.reset_index()                           # coloca índice como coluna
dt.columns = ['date','variables', 'n']          # renomeia as colunas
dt


# In[5]:


# Agrupar dataframe por data e fases da hospitalização (primeiros sintomas, internação...)
dt['variables'] = dt.variables.apply({'DT_SIN_PRI':'pri',                # primeiros sintomas
                                      'DT_INTERNA':'int',                # hospitalizado
                                      'DT_ENTUTI':'ent',                 # entrada na UTI
                                      'DT_SAIDUTI':'sai',                # saída da UTI
                                      'DT_EVOLUCA':'evo'}.get)           # alta
dt.loc[dt.n != 0, 'n'] = 1                                               # 1 se tem registro, 0 caso contrário
dt = dt.groupby(['date','variables'])['n'].sum().reset_index(name='n')   # agrupa por total de registros no dia
dt


# In[6]:


# Cria um dataframe padrão com todas dias preenchidos com todas variáveis (utilizar para concatenar bases)

# tipos e total de variáveis para utilizar como padrão
var = dt.variables.unique().tolist()     # cria uma lista com o nome das variáveis
var_unique = dt['variables'].nunique()   # número de variáveis

# cria dataframe com o intervalo (data) selecionado
date01 = '2019-01-01'                          # limite inferior (data inicial)
date02 = '2021-06-30'                          # limite superior (data final)
myd = pd.date_range(date01, date02).tolist()   # completa todos os dias dentro do limite estabelecido
myd = pd.DataFrame({'date':myd})               # cria um dataframe do intervalo selecionado
myd['variables'] = ",".join(var)               # adiciona uma coluna com o nome das variáveis

# coloca cada elemento na lista da coluna variables em uma única linha
myd = pd.DataFrame(myd.variables.str.split(',').tolist(),index=myd.date).stack()
myd = myd.reset_index([0,'date'])
myd = myd.rename(columns={'date':'date',0:'variables'})
myd


# In[7]:


# Unifica os dataframes, completando todos dias sem registro com zero
myd['date'] = pd.to_datetime(myd['date'])                                   # coloca a coluna data no padrão "YYYY-MM-DD"
dt['date'] = pd.to_datetime(dt['date'])                                     # coloca a coluna data no padrão "YYYY-MM-DD"
dfs = pd.merge(myd, dt, how="outer", on=["date","variables"])               # concatena dataframes
dfs = dfs[(dfs['date'] >= '2020-01-01') & (dfs['date'] <= '2020-12-31')]    # limite de interesse para o dataframe
dfs['n'] = dfs['n'].fillna(0)                                               # completa todos valores ausentes por zero
dfs


# In[8]:


# Salva arquivo em csv
os.chdir("C:/Users/andre/OneDrive/Pesquisas/ANDRE_TESE_COVID/TeseDados/SRAG")
dfs.to_csv('SRAG_COVID19_Y2020.csv', index=False, header = True)

