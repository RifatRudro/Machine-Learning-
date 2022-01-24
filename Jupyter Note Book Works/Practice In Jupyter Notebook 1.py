#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize

# Creating a function to examine
x = np.arange(-20, 15, 0.3)
def f(x):
    return x**2 - (5*x)/7 - 50*np.cos(x)

# Global optimization
grid = (-20, 15, 0.3)
xmin_global = optimize.brute(f, (grid, ))
print("Global minima (-20-15) at: {}".format(float(xmin_global)))

# Constrained optimization
xmin_local = optimize.fminbound(f, 5, 15)
print("Local minimum (5-15) at: {}".format(xmin_local))

# Plotting the function
fig = plt.figure(figsize=(10, 8))
plt.plot(x, f(x), 'b', label="f(x)")

# Plotting horizontal line where possible roots can be found 
plt.axhline(0, color='gray', label="Roots Level")

# Plotting the function minima
xmins = np.array([xmin_global[0], xmin_local])
plt.plot(xmins, f(xmins), 'go', label="Minima")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Finding the minimum of a function")
plt.legend(loc='best')
plt.show()


# In[2]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import os
df = pd.read_csv("F:\Searching Data/creditcard.csv")
df.head()


# In[3]:


print(df.shape)
print(df.columns)


# In[4]:


percent_missing = df.isnull().sum() * 100 / len(df)
missing_values = pd.DataFrame({'percent_missing': percent_missing})
missing_values.sort_values(by ='percent_missing' , ascending=False)


# In[5]:


figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

sns.set(style="ticks")
f = sns.countplot(x="Class", data=df, palette="bwr")
plt.show()


# In[6]:


figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
corr=df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


# In[ ]:




