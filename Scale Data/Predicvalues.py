# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:24:25 2021

@author: User
"""

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("cars2.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[23, 3.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)


********************************************
Output: 
[105.62574092] 
