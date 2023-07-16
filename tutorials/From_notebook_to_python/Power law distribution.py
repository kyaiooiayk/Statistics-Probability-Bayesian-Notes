#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Power law distribution
# 
# </font>
# </div>

# # Import modules

# In[1]:


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pareto


# # What is a log normal distribution?

# <div class="alert alert-info">
# <font color=black>
# 
# - A Power Law is a functional relationship between two quantities, where a relative change in one quantity results in a proportional relative change in the other quantity, independent of the initial size of those quantities: one quantity varies as a power of another.
# - For instance, considering the area of a square in terms of the length of its side, if the length is doubled, the area is multiplied by a factor of four.
# 
# 
# - A power law distribution has the form `Y = k Xα`, where:
#     - `X` and Y are variables of interest
#     - `α` is the law’s exponent,
#     - `k` is a constant. 
# 
#     
# - Power-law distribution is just one of many probability distributions, but it is **considered a valuable** tool to assess uncertainty issues that normal distribution cannot handle when they occur at a certain probability.
# 
# - Many processes have been found to follow power laws over substantial ranges of values. From the distribution in incomes, size of meteoroids, earthquake magnitudes, the spectral density of weight matrices in deep neural networks, word usage, number of neighbors in various networks, etc
# 
# 
# </font>
# </div>

# In[2]:


x_m = 1 #scale
alpha = [1, 2, 3] #list of values of shape parameters
plt.figure(figsize=(10,6))
samples = np.linspace(start=0, stop=5, num=1000)
for a in alpha:
    output = np.array([pareto.pdf(x=samples, b=a, loc=0, scale=x_m)])
    plt.plot(samples, output.T, label='alpha {0}' .format(a))

plt.xlabel('samples', fontsize=15)
plt.ylabel('PDF', fontsize=15)
plt.title('Probability Density function', fontsize=15)
plt.legend(loc='best')
plt.show()


# In[ ]:





# # References

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.kdnuggets.com/2021/09/advanced-statistical-concepts-data-science.html
# 
# </font>
# </div>
