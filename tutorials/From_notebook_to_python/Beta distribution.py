#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#The-connection-with-Bayes's-theorem" data-toc-modified-id="The-connection-with-Bayes's-theorem-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>The connection with Bayes's theorem</a></span></li><li><span><a href="#References" data-toc-modified-id="References-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# In[2]:


import numpy as np
from scipy.stats import beta 
import matplotlib.pyplot as plt 
import seaborn as sns


# # The connection with Bayes's theorem
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - Essentially, as α becomes larger the bulk of the probability distribution moves towards one (a coin biased to come up heads more often)
# 
# - An increase in β moves the distribution towards zero (a coin biased to come up tails more often).
# 
# - However, if both α and β increase then the distribution begins to narrow. If α and β increase equally, then the distribution will peak over θ = 0.5, which occurs when the coin is fair.
# 
# - However, perhaps the most important reason for choosing a beta distribution is because it is a conjugate prior for the Bernoulli distribution.
# 
# </font>
# </div>

# In[4]:


sns.set_palette("deep", desat=.6) 
sns.set_context(rc={"figure.figsize": (8, 4)}) 
x = np.linspace(0, 1, 100)
params = [
    (0.5, 0.5),
    (1, 1),
    (4, 3),
    (2, 5),
    (6, 6)
]


# In[6]:


for p in params:
    y = beta.pdf(x, p[0], p[1])
    plt.plot(x, y, label="$\\alpha=%s$, $\\beta=%s$" % p) 
plt.xlabel("$\\theta$, Fairness")
plt.ylabel("Density")
plt.legend(title="Parameters")
plt.show()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.quantstart.com/advanced-algorithmic-trading-ebook/ 
# - https://huyenchip.com/ml-interviews-book/contents/5.2.1.1-basic-concepts-to-review.html 
# 
# </font>
# </div>

# In[ ]:




