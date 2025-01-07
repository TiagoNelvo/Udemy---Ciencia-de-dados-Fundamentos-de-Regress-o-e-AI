import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('007-Dataset.xlsx')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(df)
df.describe()

plt.scatter(X,y, color='red')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree= 2)
X_poly = poly_reg.fit_transform(X)

print(X_poly)

regressor = LinearRegression()
regressor.fit(X_poly,y)

regressor.score(X_poly,y)

y_pred = regressor.predict(poly_reg.fit_transform(X))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y.reshape(len(y),1)),1))

print(regressor.coef_)
print(regressor.intercept_)

print(regressor.predict(poly_reg.fit_transform([[8]])))

# Plot Regression

plt.figure()
plt.scatter(X,y, color='red')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.grid(True)
plt.show()

