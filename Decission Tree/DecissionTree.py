# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:15:29 2021

@author: User
"""

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("cars2.csv")

X = df[['Weight', 'Volume']]

scaledX = scale.fit_transform(X)

print(scaledX)