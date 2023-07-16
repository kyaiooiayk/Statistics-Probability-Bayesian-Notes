#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-a-5-number-summary?" data-toc-modified-id="What-is-a-5-number-summary?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is a 5-number summary?</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Example" data-toc-modified-id="Example-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Example</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Takey 5-number summary
# 
# </font>
# </div>

# # What is a 5-number summary?
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - You have two scenarios: normal or non normal distribution.
# - The distibution is **Gaussian** and the mean plus std deviation is enough to parametrise it. 
# 
# 
# - The distribution is **not Gaussian**. In this case we can use 5 numbers to describe it:
#     - **Median**: The middle value in the sample, also called the 50th percentile or the 2nd quartile. 
#     - **1st Quartile**: The 25th percentile. 
#     - **3rd Quartile**: The 75th percentile. 
#     - **Minimum**: The smallest observation in the sample. 
#     - **Maximum**: The largest observation in the sample. 
#         
#         
# - **Parametrise** is here used as a synonym of characterise/describe.
# - Also striclty speaking quantile is not the same thing as percentile.
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[4]:


from numpy import percentile
from numpy.random import seed, rand


# # Example
# <hr style="border:2px solid black"> </hr>

# In[6]:


seed(1)
# Random samplpe
data = rand(1000)
# calculate quartiles
quartiles = percentile(data, [25, 50, 75])
# calculate min/max
data_min, data_max = data.min(), data.max() 
# display 5-number summary
print('Min: %.4f' % data_min)
print('Q1: %.4f' % quartiles[0]) 
print('Median: %.4f' % quartiles[1]) 
print('Q3: %.4f' % quartiles[2]) 
print('Max: %.4f' % data_max)


# In[ ]:




