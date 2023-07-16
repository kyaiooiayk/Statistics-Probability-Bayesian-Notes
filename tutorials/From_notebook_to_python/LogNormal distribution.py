#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** LogNormal distribution
# 
# </font>
# </div>

# # Import modules

# In[1]:


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import lognorm


# # What is a log normal distribution?

# <div class="alert alert-info">
# <font color=black>
# 
# - A log-normal distribution also known as **Galton's distribution** is a continuous probability distribution of a random variable whose logarithm is normally distributed.
# - Thus, if the random variable `X` is log-normally distributed, then `Y = ln(X)` has a normal distribution. Equivalently, if `Y` has a normal distribution, then the exponential function of Y i.e, `X = exp(Y)`, has a log-normal distribution. 
# 
# - The shape of Lognormal distribution is defined by 3 parameters:
# 
#     - σ is the shape parameter, (and is the standard deviation of the log of the distribution)
#     - θ or μ is the location parameter (and is the mean of the distribution)
#     - m is the scale parameter (and is also the median of the distribution) 
# 
# - The location and scale parameters are equivalent to the mean and standard deviation of the logarithm of the random variable as explained above.
# 
# - If x = θ, then f(x) = 0. The case where θ = 0 and m = 1 is called the standard lognormal distribution. The case where θ equals zero is called the 2-parameter lognormal distribution.
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# In[2]:


np.random.seed(42)

data = lognorm.rvs(s=0.5, loc=1, scale=1000, size=1000)

plt.figure(figsize=(10,6))
ax = plt.subplot(111)
plt.title('Generate wrandom numbers from a Log-normal distribution')
ax.hist(data, bins=np.logspace(0,5,200), density=True)
ax.set_xscale("log")

shape,loc,scale = lognorm.fit(data)

x = np.logspace(0, 5, 200)
pdf = lognorm.pdf(x, shape, loc, scale)

ax.plot(x, pdf, 'y')
plt.show()


# # References

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.kdnuggets.com/2021/09/advanced-statistical-concepts-data-science.html
# 
# </font>
# </div>
