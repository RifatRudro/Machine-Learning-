# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 18:33:13 2022

@author: User
"""

import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt

data = pd.read_csv("creditcard.csv")
data.head()

count_classes = pd.value_counts(data['Class'], sort = False)
count_classes.plot (kind='bar')
plt.title ("Class Imbalance Visualization")
plt.xlabel ("Class")
plt.ylabel ("Frequency")


