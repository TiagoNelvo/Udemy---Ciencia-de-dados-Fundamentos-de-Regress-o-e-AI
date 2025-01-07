#   Processamento de Dados
# Importando as bibliotecas

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Importando um Dataset

df = pd.read_excel('001-Dataset.xlsx')

# First look into your Dataset

df

# Dataset describe

df.describe()

# Criando algumas figuras

df.corr()

# Correlation Matrix

hm = sns.heatmap(df.corr(), annot = True)
hm.set(title = "Correlation Matrix")

# Boxplot

df.boxplot(grid = False, rot=90,fontsize=15)

# Pairplot

sns.pairplot(df)

# Parallel Coordinates Plot

column_names = list(df)

column_names

column_names[-1]

px.parallel_coordinates(df,color=column_names[-1], dimensions=column_names, title="Parallel Coordenadas Plot")

#   Limpeza de Dados(Remoção de Outliers)
# Removendo Outliers

column_names

for names in column_names[1:]:
    for x in [names]:
        q75,q25 = np.percentile(df.loc[:,x],[75,25])
        intr_qr = q75-q25
        
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
        
        df.loc[df[x] < min,x] = np.nan
        df.loc[df[x] > max,x] = np.nan


df.isnull().sum()

df_limpo = df.dropna(axis=0)
df_limpo.isnull().sum()

df_limpo

df_limpo.describe()

df_limpo.to_excel("003-Dataset_No_Outliers.xlsx")



#   Padronização de Dados
df04_a = pd.read_excel('004-Dataset(a).xlsx')
df04_b = pd.read_excel('004-Dataset(b).xlsx')

df04_a

df04_b

df04_b.describe()

df04_a.plot.kde()

min_value_a = df04_a.min()
min_value_a

max_value_a = df04_a.max()
max_value_a

std_value_a = df04_a.std()
std_value_a

mean_value_a = df04_a.mean()
mean_value_a

min_value_a[0]

df04_a.iloc[1,0]

df04_a.iloc[2,1]

x_Normalization_a = (df04_a.iloc[2,1]-min_value_a[1])/(max_value_a[1]-min_value_a[1])

x_Normalization_a

x_Standardization_a = (df04_a.iloc[0,0] - mean_value_a[0])/(std_value_a[0])

x_Standardization_a

#   Data Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df04_a)

scaler.transform(df04_a)

scalerdata = pd.DataFrame(data=scaler.transform(df04_a))

scalerdata.columns= ['X1','X2','Y']

scalerdata

scalerdata.describe()

scalerdata.plot.kde()


