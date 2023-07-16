#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** QQ plot
# 
# </font>
# </div>

# # Import modules

# In[3]:


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# # Q-Q(quantile-quantile) Plots

# <div class="alert alert-info">
# <font color=black>
# 
# - A quantile defines a particular **value** in the data set which determines how many values in a distribution are below it. 
# - **How is this plot used?** Q-Q plots can find Kurtosis (measure of tailedness) of the distribution.
# - We'll generate two plots:
#     - Using `np.random.normal` where the data will follows the theoretical quantile.
#     - Using `np.random.uniform` where you can see how the data deviate from the normal distribution as we have used an uniform one.
#     
# </font>
# </div>

# ![image.png](attachment:image.png)

# In[7]:


#create dataset with 100 values that follow a normal distribution
np.random.seed(0)
data = np.random.normal(0,1, 1000)

#view first 10 values
data[:10]

# Create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(data, line = '45')
plt.show()


# In[8]:


#create dataset of 100 UNIFORMALLY distributed values
data = np.random.uniform(0,1, 1000)

#generate Q-Q plot for the dataset
fig = sm.qqplot(data, line='45')
plt.show()


# # References

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.kdnuggets.com/2021/09/advanced-statistical-concepts-data-science.html
# 
# </font>
# </div>
