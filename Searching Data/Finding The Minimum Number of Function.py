# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:02:19 2022

@author: User
"""

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
print("Global (-20-15) at: {}".format(float(xmin_global)))

# Constrained optimization
xmin_local = optimize.fminbound(f, 5, 15)
print("Local  (5-15) at: {}".format(xmin_local))

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

