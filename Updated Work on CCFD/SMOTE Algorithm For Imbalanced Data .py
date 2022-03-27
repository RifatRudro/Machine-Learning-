#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


# In[2]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# In[5]:


credit_card = pd.read_csv('creditcard.csv')
credit_card.head(5)


# In[6]:


X = credit_card.drop('Class', axis = 1)
Y = credit_card['Class']


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.3, random_state = 80)
lr = LogisticRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print(classification_report(y_test, pred))
confu = confusion_matrix(y_test, pred)
sns.heatmap(confu, annot = True, fmt = 'd')


# In[14]:


Fraud_amount = credit_card['Amount'][credit_card['Class']==1]
print('Maximum Fraud amount:', Fraud_amount.max())
print('Minimum Fraud amount:', Fraud_amount.min())


# In[ ]:




