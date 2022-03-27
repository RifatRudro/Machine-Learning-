#!/usr/bin/env python
# coding: utf-8

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
df = pd.read_csv("creditcard.csv")
df.head()


# In[4]:


df.info()


# In[5]:


df['Class'].value_counts()


# In[ ]:





# In[ ]:




