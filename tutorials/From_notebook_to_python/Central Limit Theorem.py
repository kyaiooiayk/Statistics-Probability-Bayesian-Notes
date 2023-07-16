#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Centrol-limit-theorem-vs.-law-of-large-number" data-toc-modified-id="Centrol-limit-theorem-vs.-law-of-large-number-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Centrol limit theorem vs. law of large number</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Example-#1" data-toc-modified-id="Example-#1-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Example #1</a></span></li><li><span><a href="#Example-#2" data-toc-modified-id="Example-#2-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Example #2</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** Central limit theorem
# 
# </font>
# </div>

# # Centrol limit theorem vs. law of large number

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The theorem states that as the size of the sample increases, the distribution of the mean across multiple samples will approximate a Gaussian distribution. 
# 
# - ATTENTION! The central limit theorem is **NOT** the law of large numbers by beginners. 
# 
# - The law of large numbers states that as the size of a sample is increased, the more accurate of an estimate the sample mean will be of the population mean. 
# 
# - The central limit theorem does not state anything about a single sample mean; instead, it is broader and states something about the **distribution** of sample means. 
# 
# - **Why is this important?** Because the theorem is valid even if the original variables themselves are not normally distributed. The theorem is a key concept in probability theory because it implies that probabilistic and statistical methods that work for normal distributions can be applicable to many problems involving other types of distributions. 
#  
# 
# <br></font>
# </div>

# # Imports

# In[1]:


from numpy.random import seed
from numpy.random import randint
from numpy import mean
from matplotlib import pyplot
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
#random.seed = 42
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
sns.set_style(style='darkgrid')

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 10
mpl.rcParams['figure.dpi']= 300


# # Example #1

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Running the example creates a histogram plot of the sample means. 
# - We can tell from the shape of the distribution that the distribution  is Gaussian.
# 
# <br></font>
# </div>

# In[2]:


# seed the random number generator
seed(1)
# calculate the mean of 50 dice rolls 1000 times
means = [mean(randint(1, 7, 50)) for _ in range(1000)]
# plot the distribution of sample means
pyplot.hist(means)


# # Example #2

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - This second example takes in a dataset and check if the central limit theorem holds true. 
# 
# <br></font>
# </div>

# In[3]:


# reading the data and plotting the initial distribution
df = pd.read_csv(r'../DATASETS/toy_dataset.csv')
sns.kdeplot(df['Income'],shade=True)
print("Number of datapoints in our population: ",df.shape[0])
# population mean
population_mean = np.round(df['Income'].mean(),3)
# population std
population_std = np.round(df['Income'].std(),3)
print("Population mean is: ",population_mean)
print("Population standard deviation is: ",population_std)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Now Let's take 200 samples with each of size 100, and try to plot the distribution of their 'mean'
# 
# <br></font>
# </div>

# In[4]:


def return_mean_of_samples(total_samples,element_in_each_sample):
    """
    This function takes total samples and number of elements 
    in each sample as input and generates sample means
    """
    sample_with_n_elements_m_size = []
    for i in range(total_samples):
        sample = df.sample(element_in_each_sample).mean()['Income']
        sample_with_n_elements_m_size.append(sample)
    return (sample_with_n_elements_m_size)


# In[5]:


sample_means = return_mean_of_samples(200,100)
sns.kdeplot(sample_means,shade=True)
print("Total Samples: ",200)
print("Total elements in each sample: ",100)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Here we create various sampling distributions by varying values of total samples and elements in each samples.
# 
# <br></font>
# </div>

# In[6]:


total_samples_list = [100,500]
elements_in_each_sample_list = [50,100,500]
mean_list = []
std_list = []
key_list = []
estimate_std_list = []
key=''
pop_mean = [population_mean]*6
pop_std = [population_std]*6
for tot in total_samples_list:
    for ele in elements_in_each_sample_list:
        key = '{}_samples_with_{}_elements_each'.format(tot,ele)
        key_list.append(key)
        mean_list.append(np.round(np.mean(return_mean_of_samples(tot,ele)),3))
        std_list.append(np.round(np.array(return_mean_of_samples(tot,ele)).std(),3))
        estimate_std_list.append(np.round(population_std/(np.sqrt(ele)),3))


# In[7]:


# We summarize the results of the sampling distributons obtained
temp = pd.DataFrame(zip(key_list,pop_mean,mean_list,pop_std,estimate_std_list,std_list),columns=['Sample_Description','Population_Mean','Sample_Mean','Population_Standard_Deviation',"Pop_Std_Dev/"+u"\u221A"+"sample_size",'Sample_Standard_Deviation'])
temp


# In[8]:


def plot_distribution(sample,population_mean,i,j,color,sampling_dist_type):
    '''This function takes the sampling distribution and population mean as input and plots them together'''
    sns.kdeplot(np.array(sample),color = color,ax = axs[i,j],shade=True)
    axs[i, j].axvline(population_mean, linestyle="-", color='r', label="p_mean")
    axs[i, j].axvline(np.array(sample).mean(), linestyle="-.", color='b', label="s_mean")
    axs[i, j].set_title(key)
    axs[i, j].legend()

colors = ['r','g','b','y', 'c', 'm', 'k']
plt_grid  = [(0,0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
sample_sizes = [(100,50), (100, 100), (100, 500), (500, 50), (500, 100), (500, 500)]

total_samples_list = [100,500]
elements_in_each_sample_list = [50,100,500]

fig, axs = plt.subplots(3, 2,  figsize=(10, 9))
i = 0
for tot in total_samples_list:
    for ele in elements_in_each_sample_list:
        key = '{}_samples_with_{}_elements_each'.format(tot,ele)
        plot_distribution(return_mean_of_samples(tot,ele), population_mean , plt_grid[i][0], plt_grid[i][1] , colors[i], key)
        i = i + 1
plt.show()


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# -  Here we check if confidence intervals contains the actual population mean
# 
# <br></font>
# </div>

# In[9]:


sample_size = 50
standard_error = np.round(population_std/np.sqrt(sample_size),3)
def get_CI_percent(size):
    counter = 0
    lower_lim_ls = []
    upper_lim_ls = []
    sample_mean_ls = []
    pop_mean = [population_mean]*size
    status_ls = []
    for i in range(size):
        is_contains = False
        sample_mean = df.sample(50)['Income'].mean()
        sample_mean_ls.append(sample_mean)
        lower_lim = sample_mean - 2*standard_error
        lower_lim_ls.append(lower_lim)
        upper_lim = sample_mean + 2*standard_error
        upper_lim_ls.append(upper_lim)
        if (population_mean>=lower_lim)&(population_mean<=upper_lim):
            is_contains = True
            counter = counter + 1    
        status_ls.append(is_contains)
    print("{} % confidence Intervals contain the population mean".format(np.round(counter/size*100,2) ))
    return np.round(pd.DataFrame(zip(pop_mean,sample_mean_ls,lower_lim_ls,upper_lim_ls,status_ls),columns=['Population_Mean','Sample_Mean','Lower_Limit','Upper_Limit','Is_Present']),1)


# In[10]:


get_CI_percent(20)


# # References

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# - https://en.wikipedia.org/wiki/Central_limit_theorem
# - https://github.com/ravi207/central-limit-theorem
# 
# </font>
# </div>

# In[ ]:




