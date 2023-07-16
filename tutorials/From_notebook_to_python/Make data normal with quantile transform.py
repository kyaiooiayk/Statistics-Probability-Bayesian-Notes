#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Make data normal with quantile transform
# 
# </font>
# </div>

# # Import modules

# In[1]:


from pandas import DataFrame
from numpy import exp, mean, std
from numpy.random import randn
from sklearn.preprocessing import QuantileTransformer
from matplotlib import pyplot
from pandas import read_csv
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# # What is quantile transform?

# <div class="alert alert-info">
# <font color=black>
# 
# - **Issue**: Many machine learning algorithms prefer or perform better when numerical variables have a Gaussian or standard probability distribution
# - **Possible solution**: Quantile transforms are a technique for transforming numerical input or output variables to have a Gaussian or uniform probability distribution 
# 
# 
# - A quantile transform will map a variable’s probability distribution to another probability distribution. 
# - A quantile function, also called a percent-point function (PPF), is the inverse of the cumulative probability distribution (CDF). 
# - A CDF is a function that returns the probability of a value at or below a **given value**. 
# - The PPF is the inverse of this function and returns the value at or below a **given probability**.
# 
# </font>
# </div>

# In[2]:


# Create some skew data 
data = randn(1000)
data = exp(data)
pyplot.hist(data, bins=25)
pyplot.show()

# Apply quantile transform
data = data.reshape((len(data),1))
quantile = QuantileTransformer(output_distribution='normal') 
data_trans = quantile.fit_transform(data)
pyplot.hist(data_trans, bins=25)
pyplot.show()


# ### Sonar dataset

# In[3]:


# load dataset
dataset = read_csv('../DATASETS/sonar.csv', header=None) 
# summarize the shape of the dataset 
print(dataset.shape)
# summarize each variable
print(dataset.describe())
# histograms of the variables
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()


# ### Machine learning model on ORIGINAL dataset

# In[4]:


# load dataset
dataset = read_csv('../DATASETS/sonar.csv', header=None)
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))

# define and configure the model
model = KNeighborsClassifier()

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 

# report model performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))                                           


# ### Normal quantile transform

# In[5]:


dataset = read_csv('../DATASETS/sonar.csv', header=None)
# retrieve just the numeric input values
data = dataset.values[:, :-1]
# perform a normal quantile transform of the dataset
trans = QuantileTransformer(n_quantiles=100, output_distribution='normal') 
data = trans.fit_transform(data)
# convert the array back to a dataframe
dataset = DataFrame(data)
# histograms of the variables
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()


# <div class="alert alert-info">
# <font color=black>
# 
# - The distribution of each variable looks **normally distributed** as compared to the raw data
# 
# </font>
# </div>

# ### Machine learning model on TRANSFORMED dataset

# In[6]:


dataset = read_csv('../DATASETS/sonar.csv', header=None)
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# define the pipeline
trans = QuantileTransformer(n_quantiles=100, output_distribution='normal')
model = KNeighborsClassifier()
pipeline = Pipeline(steps=[('t', trans), ('m', model)])
# evaluate the pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
# report pipeline performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# ### Uniform Quantile Transform

# <div class="alert alert-info">
# <font color=black>
# 
# - Sometimes it can be beneficial to transform a highly exponential or multi-modal distribution to have a uniform distribution. 
# - This is especially useful for data with a large and sparse range of values, e.g. outliers that are common rather than rare. 
# - We can apply the transform by defining a QuantileTransformer class and setting the output distribution argument to ‘uniform’ (the default).  
# 
# </font>
# </div>

# In[7]:


dataset = read_csv('../DATASETS/sonar.csv', header=None)
# retrieve just the numeric input values
data = dataset.values[:, :-1]
# perform a uniform quantile transform of the dataset
trans = QuantileTransformer(n_quantiles=100, output_distribution='uniform') 
data = trans.fit_transform(data)
# convert the array back to a dataframe
dataset = DataFrame(data)
# histograms of the variables
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()


# ### Machine learning model on an uniform Quantile Transformed dataset

# In[8]:


dataset = read_csv('../DATASETS/sonar.csv', header=None)
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# define the pipeline
trans = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
model = KNeighborsClassifier()
pipeline = Pipeline(steps=[('t', trans), ('m', model)])
# evaluate the pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
# report pipeline performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# <div class="alert alert-info">
# <font color=black>
# 
# - Running the example, we can see that the uniform transform results in a lift in performance from 79.7
# percent accuracy without the transform to about 84.5 percent with the transform.
# - **Better than the normal transform** that achieved a score of 81.7 percent 
# 
# </font>
# </div>

# ### Hyperparameter tuning

# <div class="alert alert-info">
# <font color=black>
# 
# - The number of quantiles can be tuned.
# - This hyperparameter can be varied between 1-99 to see how the model behaves.
# 
# </font>
# </div>

# In[9]:


# get the dataset
def get_dataset(filename):
    # load dataset
    dataset = read_csv(filename, header=None)
    data = dataset.values
    # separate into input and output columns
    X, y = data[:, :-1], data[:, -1]
    # ensure inputs are floats and output is an integer label X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))
    return X, y

# get a list of models to evaluate
def get_models():
    models = dict()
    for i in range(1,100):
        # define the pipeline
        trans = QuantileTransformer(n_quantiles=i, output_distribution='uniform') 
        model = KNeighborsClassifier()
        models[str(i)] = Pipeline(steps=[('t', trans), ('m', model)])
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
    return scores

# define dataset
X, y = get_dataset('../DATASETS/sonar.csv')

# get the models to evaluate
models = get_models()

# evaluate the models and store results 
results = list()
for name, model in models.items():
    scores = evaluate_model(model, X, y) 
    results.append(mean(scores))
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison


# In[10]:


pyplot.plot(results)
pyplot.show()


# <div class="alert alert-info">
# <font color=black>
# 
# - A line plot is created showing the number of quantiles used in the transform versus the classification accuracy of the resulting model. 
# - We can see a bump with values less than 10 and drop and flat performance
# after that. 
# - The results highlight that there is likely some benefit in exploring different distributions and number of quantiles to see if better performance can be achieved.
# 
# </font>
# </div>

# # References

# <div class="alert alert-warning">
# <font color=black>
# 
# - Data preparation for machine learning, Jason Brownlee
# 
# </font>
# </div>

# In[ ]:




