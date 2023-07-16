#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Law-of-larger-number" data-toc-modified-id="Law-of-larger-number-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Law of larger number</a></span></li><li><span><a href="#Law-of-truly-large-number" data-toc-modified-id="Law-of-truly-large-number-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Law of truly large number</a></span></li><li><span><a href="#Generate-the-data" data-toc-modified-id="Generate-the-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Generate the data</a></span></li><li><span><a href="#Conclusions" data-toc-modified-id="Conclusions-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Conclusions</a></span></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Requirements</a></span></li></ul></div>

# # Introduction 

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Law of large number
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import numpy as np
from numpy import arange
from numpy.random import seed, randn
from numpy import mean, array
from scipy.stats import norm
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl

rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 20
mpl.rcParams['figure.dpi']= 300


# # Law of larger number

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - The law of large numbers states that the average of the results obtained from a large number of trials should be close to the expected value and will tend to become closer to the expected value as more trials are performed.
# - **Why it matters?** It guarantees stable long-term results for the averages of some **random events**. 
# - **Example:** while a casino may lose money in a single spin of the roulette wheel, its earnings will tend towards a predictable percentage over a large number of spins. 
# 
# </font>
# </div>

# # Law of truly large number

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - This is **not** the law of large number but something different! It is still related to large numbers.
# - This is the idea that when we start investigating or working with extremely large samples of observations, we increase the likelihood of seeing something strange. 
# - That by having so many samples of the underlying population distribution, the sample will contain some **astronomically rare events.** 
# 
# <br></font>
# </div>

# # Generate the data

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - If the phenomenom is truly random then we can model it with a normal curve.
# - We get the real mean from it and we use to see if the simulations converges to this value
# 
# <br></font>
# </div>

# In[2]:


xaxis = arange(30, 70, 1)
yaxis = norm.pdf(xaxis, 50, 5)
print("The mean of this distirbution is ", np.mean(xaxis))

pyplot.plot(xaxis, yaxis)
pyplot.show()


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - We can see that as we add number the mean stabilise aroung the real mean computed above
# - Generally, we can see that larger sample sizes have less error, and we would expect this trend to continue, on **average**.
# 
# <br></font>
# </div>

# In[3]:


seed(1)

# Create a list with different numbers
sizes = list()
for x in range (10, 100000, 200):
    sizes.append(x)
    
# Generate samples of different sizes and calculate their means
means = [mean(5 * randn(size) + 50) for size in sizes]

# Plot sample mean error vs sample size
pyplot.scatter(sizes, array(means))
pyplot.plot([0, 100000], [50, 50], c = "y", lw = 4)
                                  


# # Conclusions
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-danger">
# <font color=black>
# 
# - While the Law of Large Numbers is cool, it is only true to the extent its name implies: **with large sample sizes only**.
# 
# </font>
# </div>

# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://en.wikipedia.org/wiki/Law_of_large_numbers
# - Davidson-Pilon, Cameron. Bayesian methods for hackers: probabilistic programming and Bayesian inference. Addison-Wesley Professional, 2015.
#     
# </font>
# </div>

# # Requirements
# <hr style = "border:2px solid black" ></hr>

# In[4]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv')

