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

hyperparameter = {'regressor__n_neighbors': [2,3,5,10],
                  'regressor__weights': ['uniform','distance'],
                  'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],    
}


grid_search = GridSearchCV(pipe,
                           param_grid=hyperparameter,
                           return_train_score=True,
                           scoring='net_root_mean_squered_error',
                           n_jobs=-2,
                           cv=5)

grid_search.fit(X_train, y_train)

cv_best_params = grid_search.best_params_
print('\n Best hyperparameters:')
print(cv_best_params)

#   Run model with best hyperparameters

from sklearn.metrics import mean_absolute_squarred_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

pipe.set_params(**cv_best_params)

































