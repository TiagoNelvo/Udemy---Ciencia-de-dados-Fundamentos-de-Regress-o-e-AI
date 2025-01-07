import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('005-Dataset.xlsx')

df.describe()
print(df)
#   Separar o Dataset em dois Dataframes(X,y)

X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values

print(X)
print(y)

plt.figure()
plt.scatter(X,y, color = 'red')
plt.title('Experimental Data')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

#   regressão linear

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X, y)

ypred = regressor.predict(X)
#ypred = regressor.predict(y)


plt.figure()
plt.scatter(X,y, color = 'red')
plt.plot(X, ypred, color = 'blue')

# R²
regressor.score(X,y)
# Predict
regressor.predict([[0.3]])
# Coeficiente Angular
regressor.coef_
# Intercept
regressor.intercept_





