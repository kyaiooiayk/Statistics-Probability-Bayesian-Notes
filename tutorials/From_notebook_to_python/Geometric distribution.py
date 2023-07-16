#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Geometric distribution
# 
# </font>
# </div>

# # Geometric Distribution

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# 
# Geometric Distribution measures number of trials needed to get a first success in repeated Bernoulli trials. 
# So suppose we have some independent trials and each trial results in one of two possible outcomes success or failure. And our probability of success is denoted by P(success) = $p$, this number stays constant from trial to trial and $X$ represents the number of trials needed to get the first success.
# 
# For the first success to occur on the $x_{th}$ trial:
# 
# - The first $x - 1$ trial must be failures.
# - The $x_{th}$ trial must be a success.
# 
# This gives us the probability mass function of the geometric distribution:
# 
# \begin{align}
# P(X = x) = (1 - p)^{x - 1}p
# \end{align}
# 
# For the geometric distribution the minimum number $x$ can take is 1, however, there is no upper bound to that number.
# 
# Last but not least, one useful property to know about the geometric distribution is variable X's mean/expectation is:
# 
# \begin{align}
# \frac{1}{p}
# \end{align}
# 
# <br></font>
# </div>

# # Import modules

# In[2]:


from scipy.stats import geom


# # Example

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - In a large populations of adults, **30%** have received CPR training.
# - If adults from this population are randomly selected, what is the probability that the **6th** person sampled is the first that has received CPR training?
# 
# <br></font>
# </div>

# In[5]:


# Manually
p = 0.3
prob = (1 - p) ** 5 * p
print(prob)

# Scipy stats' geometric distribution
print(geom(p = p).pmf(k = 6))
print(geom(p = p).pmf(k = 1))


# In[ ]:




