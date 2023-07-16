#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Problem-#1" data-toc-modified-id="Problem-#1-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Problem #1</a></span></li><li><span><a href="#Problem-#2" data-toc-modified-id="Problem-#2-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Problem #2</a></span></li><li><span><a href="#Problem-#3" data-toc-modified-id="Problem-#3-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Problem #3</a></span></li><li><span><a href="#Problem-#4" data-toc-modified-id="Problem-#4-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Problem #4</a></span></li><li><span><a href="#Problem-#5" data-toc-modified-id="Problem-#5-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Problem #5</a></span></li><li><span><a href="#Problem-#6" data-toc-modified-id="Problem-#6-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Problem #6</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Probability problems
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


# # Problem #1
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#     
# - In a single toss of 2 fair (evenly-weighted) 6-sided dice, find the probability of that their sum will be at most 9.
# 
# - https://www.hackerrank.com/challenges/basic-probability-puzzles-1
# - https://en.wikipedia.org/wiki/Probability
#     
# 
# - There are 6 possibilities on each die. On 2 dice, there are 6 * 6 = 36 possibilities
# - There are 30 cases where sum <= 9 and we can find this in python loop.
# - The final probability will be: no_cases/no_of_possible_scenarios = 30/36
# 
# </font>
# </div>

# In[28]:


# How many possible autocome are there?
values = 6
# Probability of each outcome:
probability = 1/6

# Initialise
counter = 0

# First dice
for i in range(1, values + 1):
    # Second dice
    for j in range(1, values + 1):
        # Verify if each die will be different and their sum is 6
        if (i + j) <= 9:
            counter +=1 

# Final probability found
print("Probability: ", counter, "/", values**2, "=", counter/values**2)


# # Problem #2
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#     
# - For a single toss of 2 fair (evenly-weighted) dice, find the probability that the values rolled by each die will be different and their sum is 6.
# - https://www.hackerrank.com/challenges/basic-probability-puzzles-2
# 
# 
# - There are 6 possibilities on each die. On 2 dice, there are 6 * 6 = 36 possibilities
# - There are 11 cases where sum <= 9 and we can find this in python loop.
# - The final probability will be: no_cases/no_of_possible_scenarios = 11/36
#                                   
# </font>
# </div>

# In[30]:


# How many possible autocome are there?
values = 6
# Probability of each outcome:
probability = 1/6

# Initialise
counter = 0

# First dice
for i in range(1, values + 1):
    # Second dice
    for j in range(1, values + 1):
        # Verify if each die will be different and their sum is 6
        if i != j and i+j == 6:
            counter += 1
            
# Final probability found
print("Probability: ", counter, "/", values**2, "=", counter/values**2)         


# # Problem #3
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#     
# - There are 3 urns: X, Y and Z.
#     - Urn X contains 4 red balls and 3 black balls.
#     - Urn Y contains 5 red balls and 4 black balls.
#     - Urn Z contains 4 red balls and 4 black balls.
# - One ball is drawn from each urn. What is the probability that the 3 balls drawn consist of 2 red balls and 1 black ball?
# - https://www.hackerrank.com/challenges/basic-probability-puzzles-3
# 
# 
# - First compute the probability of red/black for each urn.
# - There are three scenarios where you have 2 red balls: RRB, RBR, BRR where you have to multiply the probabilities
# - The sum the probability of the three scenarios.
# - Which is the probability of the union of disjoint events each formed from the intersection of independent events.
#                                   
# </font>
# </div>

# In[33]:


x_prob_red = 4/7
x_prob_black = 3/7

y_prob_red = 5/9
y_prob_black = 4/9

z_prob_red = 1/2
z_prob_black = 1/2

# We have to multiply the possibilities as they are independent from each other
# keep in mind two must be red all the time
first_combination = x_prob_red * y_prob_red * z_prob_black
second_combination = x_prob_black * y_prob_red * z_prob_red
third_combination = x_prob_red * y_prob_black * z_prob_red

print(first_combination + second_combination + third_combination)


# In[32]:


(2/(3/4 * 5/9 * 1/2)) *(1/ (3/7 * 4/9 * 1/2))


# # Problem #4
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#     
# 
# - Bag1 contains 4 red balls and 5 black balls. 
# - Bag2 contains 3 red balls and 7 black balls. 
# - If one ball is drawn from the Bag1, and 2 balls are drawn from  Bag2, find the probability that 2 balls are black and 1 ball is red.
# - https://www.hackerrank.com/challenges/basic-probability-puzzles-4
# 
#                                   
# </font>
# </div>

# In[39]:


# Compute individual probability on bag1
bag1r = 4/9
bag1b = 5/9

# Compute individual probability on bag1
bag2r = 3/10
bag2b = 7/10

# Combinations
# Bag1  | Bag2  | Bag2
# black | black | red
# black | red   | black
# red   | black  | black

first_combination = bag1b * bag2b * bag2r
second_combination = bag1b * bag2r * bag2b
third_combination = bag1r * bag2b * bag2b

print(first_combination + second_combination + third_combination)


# # Problem #5
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - There are 10 people about to sit down around a round table. 
# - Find the probability that 2 particular people will sit next to one another.
# - https://www.hackerrank.com/challenges/basic-probability-puzzles-5
# 
# </font>
# </div>

# In[45]:


# One person will sit, leaving nine positions. Each position has two 
# seats, one on the left side and one on the right side. 
all_scenarios = 10-1
left_plus_right_seat = 2
left_plus_right_seat/all_scenarios 


# # Problem #6
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Bag X contains 5 white balls and 4 black balls. 
# - Bag Y contains 7 white balls and 6 black balls. 
# You draw 1 ball from bag X and, without observing its color, put it into bag Y. 
# - Now, if a ball is drawn from bag Y, find the probability that it is black.
# 
# - https://www.hackerrank.com/challenges/basic-probability-puzzles-6
# - https://www.quora.com/A-bag-contains-5-white-balls-and-4-black-balls-and-another-contains-3-white-balls-and-2-black-balls-One-ball-is-drawn-from-the-1st-bag-and-placed-unseen-in-the-2nd-bag-What-is-the-probability-that-a-ball-now-drawn
#     
# </font>
# </div>

# In[49]:


14*9


# In[53]:


# there are two scenarios
# pick a white ball + pick a black ball
# P(Y = b | X = w) * P(X = w) + P(Y = b | X = b) + P(X = b)
# probability of chosing black from Y given that we draw a white from X 
# plus
# probability of chosing black from Y given that we draw a black from X 
# (6/14 * 5/9) + (7=(6+1)/14 * 4/9) = 29/63 = 0.46031746

print((6/14 * 5/9) + (7/14 * 4/9))


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




