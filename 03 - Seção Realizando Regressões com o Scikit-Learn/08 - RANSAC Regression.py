import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('008-Dataset.xlsx')
df
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

plt.scatter(X,y, color='red')
plt.title('Experimental Data')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, y)

ypred = regressor.predict(X)

plt.figure()
plt.scatter(X,y, color='red')
plt.plot(X, ypred, color = 'blue')

#   Makeing a single prediction

regressor.predict([[2.5]])

# RANSAC

plt.scatter(X, y, color='yellowgreen', marker=".")
plt.title("Experimental Data")
plt.xlabel('X')
plt.ylabel('y')
plt.show()

from sklearn.linear_model import RANSACRegressor

regressor = RANSACRegressor()
regressor.fit(X,y)

regressor.score(X,y)

# Plot Regression

plt.scatter(X,y, color="yellowgreen",marker=".")
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Ransac Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.figure()

#   Making a single prediction

regressor.predict([[2.5]])


