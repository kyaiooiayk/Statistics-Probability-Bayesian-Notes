#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Continous-and-discrete-distrubutions" data-toc-modified-id="Continous-and-discrete-distrubutions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Continous and discrete distrubutions</a></span></li><li><span><a href="#Gaussian-distribution" data-toc-modified-id="Gaussian-distribution-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Gaussian distribution</a></span></li><li><span><a href="#Lognormal-distribution" data-toc-modified-id="Lognormal-distribution-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Lognormal distribution</a></span></li><li><span><a href="#Poisson-distribution" data-toc-modified-id="Poisson-distribution-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Poisson distribution</a></span></li><li><span><a href="#Exponential-distribution" data-toc-modified-id="Exponential-distribution-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Exponential distribution</a></span></li><li><span><a href="#Binomial-distribution" data-toc-modified-id="Binomial-distribution-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Binomial distribution</a></span></li><li><span><a href="#Student’s-t-distribution" data-toc-modified-id="Student’s-t-distribution-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Student’s t distribution</a></span></li><li><span><a href="#References" data-toc-modified-id="References-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** 7 most important distributions
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# # Continous and discrete distrubutions
# <hr style = "border:2px solid black" ></hr>

# In[3]:


# for continuous
a = 0
b = 50
size = 5000

X_continuous = np.linspace(a, b, size)
continuous_uniform = stats.uniform(loc=a, scale=b)
continuous_uniform_pdf = continuous_uniform.pdf(X_continuous)

# for discrete
X_discrete = np.arange(1, 7)
discrete_uniform = stats.randint(1, 7)
discrete_uniform_pmf = discrete_uniform.pmf(X_discrete)

# plot both tables
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
# discrete plot
ax[0].bar(X_discrete, discrete_uniform_pmf)
ax[0].set_xlabel("X")
ax[0].set_ylabel("Probability")
ax[0].set_title("Discrete Uniform Distribution")

# continuous plot
ax[1].plot(X_continuous, continuous_uniform_pdf)
ax[1].set_xlabel("X")
ax[1].set_ylabel("Probability")
ax[1].set_title("Continuous Uniform Distribution")
plt.show()


# # Gaussian distribution
# <hr style = "border:2px solid black" ></hr>

# In[4]:


mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.subplots(figsize=(8, 5))
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.title("Normal Distribution")
plt.show()


# # Lognormal distribution
# <hr style = "border:2px solid black" ></hr>

# In[5]:


X = np.linspace(0, 6, 500)

std = 1
mean = 0
lognorm_distribution = stats.lognorm([std], loc=mean)
lognorm_distribution_pdf = lognorm_distribution.pdf(X)

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(X, lognorm_distribution_pdf, label="μ=0, σ=1")
ax.set_xticks(np.arange(min(X), max(X)))

std = 0.5
mean = 0
lognorm_distribution = stats.lognorm([std], loc=mean)
lognorm_distribution_pdf = lognorm_distribution.pdf(X)
plt.plot(X, lognorm_distribution_pdf, label="μ=0, σ=0.5")

std = 1.5
mean = 1
lognorm_distribution = stats.lognorm([std], loc=mean)
lognorm_distribution_pdf = lognorm_distribution.pdf(X)
plt.plot(X, lognorm_distribution_pdf, label="μ=1, σ=1.5")

plt.title("Lognormal Distribution")
plt.legend()
plt.show()


# # Poisson distribution
# <hr style = "border:2px solid black" ></hr>

# In[6]:


print(stats.poisson.pmf(k=9, mu=3))


# In[8]:


# generate random values from poisson distribution with sample size of 500
X = stats.poisson.rvs(mu=3, size=500)

plt.subplots(figsize=(8, 5))
plt.hist(X, density=True, edgecolor="black")
plt.title("Poisson Distribution")
plt.show()


# # Exponential distribution
# <hr style = "border:2px solid black" ></hr>

# In[9]:


X = np.linspace(0, 5, 5000)

exponetial_distribtuion = stats.expon.pdf(X, loc=0, scale=1)

plt.subplots(figsize=(8, 5))
plt.plot(X, exponetial_distribtuion)
plt.title("Exponential Distribution")
plt.show()


# # Binomial distribution
# <hr style = "border:2px solid black" ></hr>

# In[10]:


# Binomial
X = np.random.binomial(n=1, p=0.5, size=1000)

plt.subplots(figsize=(8, 5))
plt.hist(X)
plt.title("Binomial Distribution")
plt.show()


# # Student’s t distribution
# <hr style = "border:2px solid black" ></hr>

# In[13]:


X1 = stats.t.rvs(df=1, size=4)
X2 = stats.t.rvs(df=3, size=4)
X3 = stats.t.rvs(df=9, size=4)

plt.subplots(figsize=(8, 5))
sns.kdeplot(X1, label="1 d.o.f")
sns.kdeplot(X2, label="3 d.o.f")
sns.kdeplot(X3, label="6 d.o.f")
plt.title("Student's t distribution")
plt.legend()
plt.show()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://towardsdatascience.com/probability-distributions-to-be-aware-of-for-data-science-with-code-c4a6bb8b0e9a
# 
# </font>
# </div>

# In[ ]:




