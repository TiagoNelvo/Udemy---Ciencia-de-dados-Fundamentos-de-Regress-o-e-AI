import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('006-Dataset.xlsx')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

df.describe()
print(df)
# Separar dados que serão utilizados para treinar o Machine Learning e dados que serão utilzados para testar a regressão

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

# quadrado
regressor.score(X_test,y_test)
#intercept
regressor.intercept_
# coeficiente angular
regressor.coef_
#predict 25 anos e 5 anos de estudo
regressor.predict([[24,5]])

y_pred = regressor.predict(X_test)
plt.scatter(y_test,y_pred, color = 'red')
plt.plot(y_test,y_test, color = 'blue')








