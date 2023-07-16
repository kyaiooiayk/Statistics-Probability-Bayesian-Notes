#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** How to test for Normality and Homoscedasticity
# 
# <br></font>
# </div>

# # Goal of this notebook

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# 1. <b>Understand the problem</b>. We'll look at each variable and understand their meaning and importance for this problem.
# 2. <b>Univariable study</b>. We'll just focus on the dependent variable ('SalePrice') and try to know a little bit more about it.
# 3. <b>Multivariate study</b>. We'll try to understand how the dependent variable and independent variables relate.
# 4. <b>Basic cleaning</b>. We'll clean the dataset and handle the missing data, outliers and categorical variables.
# 5. <b>Test assumptions</b>. We'll check if our data meets the assumptions required by most multivariate techniques.
# 
# <br></font>
# </div>

# # Import modules

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the dataset

# In[2]:


df_train = pd.read_csv('../DATASETS/House_Prices_Advanced_Regression_Techniques/train.csv')


# In[3]:


# Get the columns' name
df_train.columns


# # First things first: analysing 'SalePrice'

# In[4]:


# Descriptive statistics summary
df_train['SalePrice'].describe()


# In[5]:


# Histogram
sns.distplot(df_train['SalePrice']);


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Deviate from the normal distribution.
# - Have appreciable positive skewness.
# - Show peakedness. 
# 
# <br></font>
# </div>

# In[6]:


# Check for skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# In[7]:


# Scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[8]:


# Scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# ## Relationship with categorical features

# In[9]:


# Box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[10]:


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Both relationships are positive, which means that as one variable increases, the other also increases. In the case of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.
# - 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. The relationship seems to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase with the overall quality.
# - The trick here seems to be the choice of the right features (**feature selection**) and not the definition of complex relationships between them (**feature engineering**).
# 
# <br></font>
# </div>

# # Correlation matrix

# In[11]:


corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The idea is that if two variables are stongly corellated they may be collinear.
# - Collinearity tells us the two variables brings the same sort of iformation to the model.
# - If this is the case then we can get rid of one of them as the other will be able to provide the same level of info.
# - Let's find the most 10 most correlated variables to **SalePrice** 
# 
# <br></font>
# </div>

# In[12]:


# Saleprice correlation matrix - zoomed heatmap style
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - This mega scatter plot gives us a reasonable idea about variables relationships. 
# - This allows us to see what sort of relationship the data have.
# 
# <br></font>
# </div>

# In[13]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# # Missing data

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# Important questions when thinking about missing data:
# * How prevalent is the missing data?
# * Is missing data random or does it have a **pattern**? 
# 
# <br></font>
# </div>

# In[14]:


# Missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - We'll consider that when more than **15%** of the data is missing, we should delete the corresponding variable and pretend it never existed. This means that we will not try any trick to fill the missing data in these cases. According to this, there is a set of variables (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.) that we should delete. The point is: will we miss this data? I don't think so. None of these variables seem to be very important, since most of them are not aspects in which we think about when buying a house (maybe that's the reason why data is missing?). Moreover, looking closer at the variables, we could say that variables like 'PoolQC', 'MiscFeature' and 'FireplaceQu' are strong candidates for **outliers**, so we'll be happy to delete them.
# - We have one missing observation in 'Electrical'. Since it is just one observation, we'll delete this observation and keep the variable.
# 
# <br></font>
# </div>

# In[15]:


# Dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# Just checking that there's no missing data missing
df_train.isnull().sum().max() 


# # Outliers

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Outliers can markedly affect our models and can be a **valuable source of information**, providing us insights about specific behaviours.
# - Here, we'll just do a quick analysis through the standard deviation of 'SalePrice' and a set of scatter plots.
# - We'use both uni- and multivariate analysis, essentially, looking at one feature only and then compariong it against another one.
# 
# <br></font>
# </div>

# ## Univariate analysis

# The primary concern here is to establish a threshold that defines an observation as an outlier. To do so, we'll standardize the data. In this context, data standardization means converting data values to have mean of 0 and a standard deviation of 1.

# In[16]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# How 'SalePrice' looks with her new clothes:
# 
# * Low range values are similar and not too far from 0.
# * High range values are far from 0 and the 7.something values are really out of range.
# 
# For now, we'll not consider any of these values as an outlier but we should be careful with those two 7.something values.

# ## Bivariate analysis

# We already know the following scatter plots by heart. However, when we look to things from a new perspective, there's always something to discover. 

# In[17]:


#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# What has been revealed:
# 
# * The two values with bigger 'GrLivArea' seem strange and they are not following the crowd. We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price. I'm not sure about this but I'm quite confident that these two points are not representative of the typical case. Therefore, we'll define them as outliers and delete them.
# * The two observations in the top of the plot are those 7.something observations that we said we should be careful about. They look like two special cases, however they seem to be following the trend. For that reason, we will keep them.

# In[18]:


#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# In[19]:


#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# We can feel tempted to eliminate some observations (e.g. TotalBsmtSF > 3000) but I suppose it's not worth it. We can live with that, so we'll not do anything.

# # Getting hard core

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# 4 assumptions should be tested:
# 
# * <b>Normality</b> - When we talk about normality what we mean is that the data should look like a normal distribution. This is important because several statistic tests rely  on this (e.g. t-statistics). In this exercise we'll just check univariate normality for 'SalePrice' (which is a limited approach). Remember that univariate normality doesn't ensure multivariate normality (which is what we would like to have), but it helps. Another detail to take into account is that in big samples (>200 observations) normality is not such an issue. However, if we solve normality, we avoid a lot of other problems (e.g. heteroscedacity) so that's the main reason why we are doing this analysis.
# 
# * <b>Homoscedasticity</b> - refers to the assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.
# 
# * <b>Linearity</b>- The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations. However, we'll not get into this because most of the scatter plots we've seen appear to have linear relationships.
# 
# * <b>Absence of correlated errors</b> - Correlated errors, like the definition suggests, happen when one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related. We'll also not get into this. However, if you detect something, try to add a variable that can explain the effect you're getting. That's the most common solution for correlated errors.
# 
# 
# <br></font>
# </div>

# ## Normality test

# In[ ]:


<div class="alert alert-block alert-info">
<font color=black><br>

- We'll check for normality in two ways: 
- via Histogram
- via Normal probability plot where data distribution should closely follow the diagonal that represents the normal distribution.

<br></font>
</div>


# In[20]:


# Histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - 'SalePrice' is not normal. 
# - A simple log transformation can solve the problem. 
# 
# <br></font>
# </div>

# In[21]:


# Applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[22]:


# Transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[23]:


# Histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[24]:


#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


# In[25]:


#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[26]:


#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - **Issue**: A significant number of observations with value zero. This **doesn't allow** us to do log transformations.
# - Having value zero means the house does not have a basement.
# - **Suggestion**: To apply a log transformation here, **we'll create** a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.
#     
# <br></font>
# </div>

# In[27]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[28]:


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[29]:


#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# ## Homoscedasticity test

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - The best approach to test homoscedasticity for two metric variables is graphically. 
# - Departures from an equal dispersion are shown by such shapes as **cones** or **diamonds**.
# 
# <br></font>
# </div>

# In[30]:


#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Older versions of this scatter plot (previous to log transformations), had a conic shape (go back and check 'Scatter plots between 'SalePrice' and correlated variables (move like Jagger style)'). 
# - As you can see, the current scatter plot doesn't have a conic shape anymore. That's the power of normality! 
# - Just by ensuring normality in some variables, we solved the homoscedasticity problem. 
# 
# <br></font>
# </div>

# In[31]:


# Scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);


# We can say that, in general, 'SalePrice' exhibit equal levels of variance across the range of 'TotalBsmtSF'. Cool!

# # References

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - [Link to code](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
# - [Hair et al., 2013, Multivariate Data Analysis, 7th Edition](https://amzn.to/2JuDmvo)
# 
# <br></font>
# </div>

# # Conclusion

# <div class="alert alert-block alert-danger">
# <font color=black><br>
# 
# Nice discussion on how to test
# - for normality.
# - for Homoscedasticity.
# 
# <br></font>
# </div>

# In[ ]:




