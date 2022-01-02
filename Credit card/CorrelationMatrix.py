# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 18:53:10 2022

@author: User
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

 
# create dataframe from file
dataframe = pd.read_csv("creditcard.csv")
 
# show dataframe
print(dataframe)
 
# use corr() method on dataframe to
# make correlation matrix
matrix = dataframe.corr()
 
# print correlation matrix
print("Correlation Matrix is : ")
print(matrix)
sn.heatmap(matrix, annot=True)
plt.show()


