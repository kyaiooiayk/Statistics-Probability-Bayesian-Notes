#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Poisson distribution
# 
# </font>
# </div>

# # Import modules

# In[11]:


import random
from numpy.random import poisson
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# # Poisson distribution

# <div class="alert alert-info">
# <font color=black>
# 
# - In probability theory and statistics, the Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event.
# - In very simple terms, A Poisson distribution can be used to estimate how likely it is that something will happen "X" number of times.
# - Some examples of Poisson processes are customers calling a help center, radioactive decay in atoms, visitors to a website, photons arriving at a space telescope, and movements in a stock price. Poisson processes are usually associated with time, but they do not have to be. 
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# In[12]:


lam_list = [1, 4, 9] #list of Lambda values  

plt.figure(figsize=(10,6))
samples = np.linspace(start=0, stop=5, num=1000)

for lam in lam_list:
    sns.distplot(poisson(lam=lam, size=10), hist=False, label='lambda {0}'.format(lam))

plt.xlabel('Poisson Distribution', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.legend(loc='best')
plt.show()


# # References

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.kdnuggets.com/2021/09/advanced-statistical-concepts-data-science.html
# - [numpy.poisson](https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html)
#     
# </font>
# </div>

# In[ ]:




