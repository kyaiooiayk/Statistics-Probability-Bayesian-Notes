#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-an-independence-test?" data-toc-modified-id="What-is-an-independence-test?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is an independence test?</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Create-a-contigency-table" data-toc-modified-id="Create-a-contigency-table-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Create a contigency table</a></span></li><li><span><a href="#Run-the-test" data-toc-modified-id="Run-the-test-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Run the test</a></span></li><li><span><a href="#Post-process-the-results" data-toc-modified-id="Post-process-the-results-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Post-process the results</a></span></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Independence test
# 
# </font>
# </div>

# # What is an independence test?
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Statistical tests allows you to determine whether the output variable is dependentor independent of the input variables. 
# 
# - If **independent**, then the input variable is a candidate for a feature that may be **irrelevant** to the problem and removed from the dataset. 
# 
# - The Pearson’s **Chi-Squared** statistical hypothesis is an example of a test for independence between **categorical** variables.
# 
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[6]:


from scipy.stats import chi2
from scipy.stats import chi2_contingency
from IPython.display import Markdown, display


# # Create a contigency table
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - **What is a contigency table?** It is table which intent is to help determine whether one variable is contingent upon or depends upon the other variable. 
# - For example, does an interest in math or science depend on gender, or are they independent? 
# <br>
# - A contingency table is defined below that has a different number of observations for each population (row), but a similar proportion across each group (column). 
# - Given the similar proportions, we would expect the test to find that the groups are similar and that the variables are independent (fail to reject the null hypothesis, or H0)
# 
# </font>
# </div>

# In[7]:


table = [ [10, 20, 30], [6, 9, 17]]
table


# # Run the test
# <hr style="border:2px solid black"> </hr>

# In[9]:


stat, p, dof, expected = chi2_contingency(table) 
print('dof=%d' % dof)
print(expected)


# # Post-process the results
# <hr style="border:2px solid black"> </hr>

# In[10]:


# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat)) 

if abs(stat) >= critical:
    print('Dependent (reject H0)') 
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p)) 

if p <= alpha:
    print('Dependent (reject H0)') 
else:
    print('Independent (fail to reject H0)')


# # References
# <hr style="border:2px solid black"> </hr>

# 
# <div class="alert alert-warning">
# <font color=black>
# 
# - https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
# 
# </font>
# </div>

# In[ ]:




