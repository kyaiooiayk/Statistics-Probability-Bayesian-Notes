#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Gaussian-(normal-distribution)" data-toc-modified-id="Gaussian-(normal-distribution)-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Gaussian (normal distribution)</a></span></li><li><span><a href="#CDF-=-Cumulative-Density-function" data-toc-modified-id="CDF-=-Cumulative-Density-function-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>CDF = Cumulative Density function</a></span></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** PDF and CDF
# 
# </font>
# </div>

# # Imports

# In[1]:


from numpy import arange
from scipy.stats import norm
from matplotlib import pyplot


# # Gaussian (normal distribution)

# <div class="alert alert-info">
# <font color=black>
# 
# - Creates a plot showing the sample space in the x-axis and the likelihood of each value of the y-axis.
# - This plot is called **PDF** = Probability Density Function.
# - The normal distribution is parameterised using the mean and standard deviation only
# 
# </font>
# </div>

# In[8]:


sample_space = arange(-8, 8, 0.001)
mean = 0.0
stdev = 2.0
# Calculate the pdf
pdf = norm.pdf(sample_space, mean, stdev)
#Pplot
pyplot.plot(sample_space, pdf)
pyplot.show()


# # CDF = Cumulative Density function

# <div class="alert alert-info">
# <font color=black>
# 
# - Running the example creates a plot showing an S-shape with the sample space on the x-axis and the cumulative probability of the y-axis. 
# - We can see that a value of 2 covers close to 100% of the observations, with only a very thin tail of the distribution beyond that point. This related about the 1, 2 and 3 sigma standard deviation which tells us that 68, 95, adn 99.7% of the onbservation fall in that range.
# - We can also see that the mean value of zero shows 50% of the observations before and after that point. In fact the normal distribution is symmetric.
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# In[9]:


cdf = norm.cdf(sample_space)
pyplot.plot(sample_space, cdf)
pyplot.show()


# # References

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://itrevolution.com/whats-your-sigma/
# 
# </font>
# </div>

# In[ ]:




