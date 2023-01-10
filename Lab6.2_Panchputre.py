#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author - Komal Panchputre

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plot

# Import The Data From YAHOO On Historical Value Of The Stock Of DISNEY
dis_stock_data = pd.read_csv("DIS.csv")
print('Raw Disney Data From Yahoo Finance : ')
print(data.head())


# Remove Date And Adj Close Columns As They Are Not Needed For PLOT
dis_stock_data = dis_stock_data.drop('Date',axis=1) 
dis_stock_data = dis_stock_data.drop('Adj Close',axis = 1)
print('\n\nData After Removing Date and Adj Close : ')
print(data.head())

# Split Into Train And Test Data
data_X = dis_stock_data.loc[:,dis_stock_data.columns !=  'Close' ]
data_Y = dis_stock_data['Close']

train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, test_size = 0.25)
print('\n\nTraining Set IS - ')
print(train_X.head())
print(train_y.head())

# Creating The Regressor
lin_reg = LinearRegression()
lin_reg.fit(train_X, train_y)


# Make Predictions And Evaluate The Results
predict_y = lin_reg.predict(test_X)
print('Prediction Score : ' , lin_reg.score(test_X, test_y))

error = mean_squared_error(test_y, predict_y)
print('Mean Squared Error : ', error)


# Plot The Predicted And The Expected Values
figure = plot.figure()
ax = plot.axes()
ax.grid()
ax.set(xlabel='Close ($)',ylabel='Open ($)', title='DISNEY Stock Prediction Using Linear Regression')
ax.plot(test_X['Open'], test_y)
ax.plot(test_X['Open'], predict_y)
figure.savefig('DIS_LIN_REGRESSION.png')
plot.show()

