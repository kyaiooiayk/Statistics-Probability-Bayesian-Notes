#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-it?" data-toc-modified-id="What-is-it?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is it?</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Import-dataset" data-toc-modified-id="Import-dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import dataset</a></span></li><li><span><a href="#Run-T-test-Using-Equal-Sample-Size" data-toc-modified-id="Run-T-test-Using-Equal-Sample-Size-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Run T-test Using Equal Sample Size</a></span></li><li><span><a href="#Run-T-test-Using-Unequal-Sample-Size" data-toc-modified-id="Run-T-test-Using-Unequal-Sample-Size-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Run T-test Using Unequal Sample Size</a></span></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** T-test
# 
# </font>
# </div>

# # What is it?
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - The t-test is a statistical test that can be used to determine if there is a significant difference between the means of two independent samples of data. 
# - In this tutorial, we illustrate the most basic version of the t-test, for **which it is assumed** that the two samples have equal variances.
# - Other advanced versions of the t-test include the **Welch's t-test**, which is an adaptation of the t-test, and is more reliable when the two samples have unequal variances and possibly unequal sample sizes.
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[1]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# # Import dataset
# <hr style="border:2px solid black"> </hr>

# In[2]:


from sklearn import datasets
iris = datasets.load_iris()
sep_length = iris.data[:, 0]
a_1, a_2 = train_test_split(sep_length, test_size=0.4, random_state=0)
b_1, b_2 = train_test_split(sep_length, test_size=0.4, random_state=1)


# # Run T-test Using Equal Sample Size
# <hr style="border:2px solid black"> </hr>

# <div class = "alert alert-info" >
# <font color = black >
# 
# - We observe that the using “true” or “false” for the “equal-var” parameter does not change the t-test results that much.
# - We also observe that interchanging the order of the sample arrays a_1 and b_1 yields a negative t-test value, but does not change the magnitude of the t-test value, as expected.
# - Since the calculated p-value is way larger than the threshold value of 0.05, we can reject the null hypothesis that the difference between the means of sample 1 and sample 2 are significant. 
# - This shows that the sepal lengths for sample 1 and sample 2 were drawn from same population data.
# 
# </font >
# </div >

# In[3]:


# Calculate the sample means and sample variances

mu1 = np.mean(a_1)
mu2 = np.mean(b_1)

np.std(a_1)
np.std(b_1)


# In[4]:


# Implement t-test
stats.ttest_ind(a_1, b_1, equal_var=False)


# In[5]:


stats.ttest_ind(b_1, a_1, equal_var=False)


# In[6]:


stats.ttest_ind(a_1, b_1, equal_var=True)


# # Run T-test Using Unequal Sample Size
# <hr style="border:2px solid black"> </hr>

# In[7]:


a_1, a_2 = train_test_split(sep_length, test_size=0.4, random_state=0)
b_1, b_2 = train_test_split(sep_length, test_size=0.5, random_state=1)


# In[8]:


# Calculate the sample means and sample variances

mu1 = np.mean(a_1)
mu2 = np.mean(b_1)

np.std(a_1)
np.std(b_1)


# In[9]:


stats.ttest_ind(a_1, b_1, equal_var=False)


# # References
# <hr style="border:2px solid black"> </hr>

# 
# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.kdnuggets.com/2023/01/performing-ttest-python.html
# 
# </font>
# </div>

# In[ ]:




