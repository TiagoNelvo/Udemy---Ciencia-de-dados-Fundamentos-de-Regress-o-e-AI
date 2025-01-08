import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#   Set Default Parameters - To make this notebook's output stable across runs

default_test_size = 0.2

seed = 42

# Importing the dataset

df = pd.read_excel('009-Dataset.xlsx')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

df 
df.describe()

# Create Pipeline with a StandardScaler and a Regressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=default_test_size, random_state=seed)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', KNeighborsRegressor())
])

#   Grid Search with Cross Validation

hyperparameters = {'regressor__n_neighbors': [2,3,5,10],
                  'regressor__weights': ['uniform','distance'],
                  'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],    
}

grid_search = GridSearchCV(pipe, 
                           param_grid=hyperparameters, 
                           return_train_score=True, 
                           scoring='neg_root_mean_squared_error',
                           n_jobs=-2,
                           cv = 5)


grid_search.fit(X_train, y_train)

cv_best_params = grid_search.best_params_
print('\n Best hyperparameters:')
print(cv_best_params)

#   Run model with best hyperparameters

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

pipe.set_params(**cv_best_params)

pipe.get_params()

pipe.fit(X_train, y_train)

y_test_pred = pipe.predict(X_test)


#   Analysis of Regression Errors


# Análise dos erros das previsões

rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)


df_metricas = pd.DataFrame(data={'RSME':[rmse_test], 'MAE':[mae_test], 'MAPE':[mape_test], 'R²':[r2_test]})

df_metricas

#   Plot Results

y_pred = pd.DataFrame(data=pipe.predict(X_test), columns=['Predicted Valeues'])

y_real = pd.DataFrame(data=y_test, columns=['Real Values'])

# Prepartion of the comparitive DataFrame between the prediction and the actual value

df_comparison = pd.concat([y_real, y_pred],axis=1)
df_comparison.columns = ['Real_Data','Predicted_Value']
df_comparison['Percentage_difference'] = 100*(df_comparison['Predicted_Value'] - df_comparison['Real_Data'])/df_comparison['Real_Data']
df_comparison['Average'] = df_comparison['Real_Data'].mean()
df_comparison['Q1'] = df_comparison['Real_Data'].quantile(0.25)
df_comparison['Q3'] = df_comparison['Real_Data'].quantile(0.75)
df_comparison['USL'] = df_comparison['Real_Data'].mean() + 2*df_comparison['Real_Data'].std()
df_comparison['LSL'] = df_comparison['Real_Data'].mean() - 2*df_comparison['Real_Data'].std()

df_comparison.sort_index(inplace=True)

df_comparison

# Graphic visualization of predictions by real values

def grafico_real():
    plt.figure(figsize=(25,10))
    plt.title('Real Value vs Predicted Value', fontsize=25)
    plt.plot(df_comparison.index, df_comparison['Real_Data'], label = 'Real', marker='D', markersize=10, linewidth=0)
    plt.plot(df_comparison.index, df_comparison['Predicted_Value'], label = 'Predicted', c='r', linewidth=1.5)
    plt.plot(df_comparison.index, df_comparison['Average'], label = 'Mean', linestyle='dashed', c='yellow')
    plt.plot(df_comparison.index, df_comparison['Q1'], label = 'Q1', linestyle='dashed',c='g')
    plt.plot(df_comparison.index, df_comparison['Q3'], label = 'Q3', linestyle='dashed',c='g')

    plt.plot(df_comparison.index, df_comparison['USL'], label = 'USL', linestyle='dashed',c='r')
    plt.plot(df_comparison.index, df_comparison['LSL'], label = 'LSL', linestyle='dashed',c='r')

    plt.legend(loc='best')
    plt.legend(fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

grafico_real()

def grafico_real_scatterplot():
    plt.figure(figsize=(25,10))
    plt.title('Real Value vs Predicted Value',fontsize=25)
    plt.scatter(df_comparison['Real_Data'], df_comparison['Predicted_Value'], s=100)
    plt.plot(df_comparison['Real_Data'],df_comparison['Real_Data'],c='r')

    plt.xlabel('Real', fontsize=25)
    plt.ylabel('Predicted', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

grafico_real_scatterplot()





