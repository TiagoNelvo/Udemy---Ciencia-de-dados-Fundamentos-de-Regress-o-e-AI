import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Set Default Parameters

default_test_size = 0.2

seed = 42

#Importing the datase

df = pd.read_excel('010-Dataset.xlsx')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

df

# Create Pipeline with a StandardScaler and a Regressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

# SVR:   
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=default_test_size, random_state=seed)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', SVR())
])

# Grid Search with Cross Validation

#hyperparameters grid to search within
hyperparameters = {'regressor__kernel': ['linear', 'poly','rbf','sigmoid'],
                   'regressor__degree': [x for x in range(2,10,1)],
                   'regressor__tol': [1e-4],
                   'regressor__c': [0.5,1,2,3,5],
                   'regressor__epsilon': [0.01,0.05,0.1,0.5,1],             
                   }


grid_search = GridSearchCV(pipe,
                           param_grid=hyperparameters,
                           return_train_score=True,
                           scoring='neg_root_mean_squared_error',
                           n_jobs=-2,
                           cv = 5)

grid_search.fit(X_train,y_train)

