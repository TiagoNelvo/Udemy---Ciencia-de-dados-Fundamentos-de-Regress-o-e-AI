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











