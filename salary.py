# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:17:42 2020

@author: Mudhurika
"""
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets 
dataset = pd.read_csv('Salary_Data.csv')
X =  dataset.iloc[:,0:1].values
y  = dataset.iloc[:,1].values

#applying feature scaling on both x and y 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

#Train the SVR model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Predicting values accordingly
y_pred = regressor.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)

#visualize the plot 
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') #orignal
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)), color = 'blue') #predicted
plt.title('Years VS Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#high resolution plot 
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
