#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Import-modules" data-toc-modified-id="Import-modules-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import modules</a></span></li><li><span><a href="#Generate-some-data" data-toc-modified-id="Generate-some-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Generate some data</a></span></li><li><span><a href="#Significance-test" data-toc-modified-id="Significance-test-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Significance test</a></span><ul class="toc-item"><li><span><a href="#T-Test" data-toc-modified-id="T-Test-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>T-Test</a></span></li><li><span><a href="#ANOVA" data-toc-modified-id="ANOVA-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>ANOVA</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Statistical significance tests
# 
# </font>
# </div>

# # Import modules

# In[3]:


from numpy import std
from numpy import mean
from numpy.random import seed
from numpy.random import randn
from scipy.stats import f_oneway
from scipy.stats import ttest_ind


# # Generate some data

# <div class="alert alert-info">
# <font color=black>
# 
# - Generate two **DIFFERENT** sets of univariate observations  
# 
# </font>
# </div>

# In[4]:


seed(71)

data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51

print('data1: mean = %.3f stdv = %.3f' % (mean(data1), std(data1)))
print('data2: mean = %.3f stdv = %.3f' % (mean(data2), std(data2)))


# # Significance test

# <div class="alert alert-info">
# <font color=black>
# 
# - We expect the statistical tests to discover that the samples were drawn from **differing** distributions.
# - Consider that the small sample size of 100 observations per sample will add some noise to this decision.  
# - Essentially we are checking if the have the same mean
# 
# </font>
# </div>

# ## T-Test

# In[5]:


# USE T-TEST TO CHECK IF THEY HAVE THE SAME MEAN

# Compare samples
stat, p = ttest_ind(data1, data2) 
print('Statistics=%.3f, p=%.3f' % (stat, p)) 

# Interpret the result
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')


# ## ANOVA 

# In[7]:


# USE TO CHECK IF THEY HAVE THE SAME MEAN
seed(71)

# generate three independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 50
data3 = 5 * randn(100) + 53

# compare samples
stat, p = f_oneway(data1, data2, data3) 
print('Statistics=%.3f, p=%.3f' % (stat, p)) # interpret

alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)') 
else:
    print('Different distributions (reject H0)')


# # References

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://en.wikipedia.org/wiki/Statistical_hypothesis_testing 
# 
# </font>
# </div>

# In[ ]:




