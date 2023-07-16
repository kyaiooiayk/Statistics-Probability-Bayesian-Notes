#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Bayesian-Approach-and-Markov-chain" data-toc-modified-id="Bayesian-Approach-and-Markov-chain-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Bayesian Approach and Markov chain</a></span></li><li><span><a href="#References" data-toc-modified-id="References-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Simple example of a Markov chain 
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

import quantecon as qe
from quantecon import MarkovChain
import networkx as nx
from pprint import pprint 

import pyflux as pf
from scipy.stats import kurtosis


# # Bayesian Approach and Markov chain
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - The way we approach probability is of central importance in the sense that it distin‐ guishes the classical (or Frequentist) and Bayesian approaches. 
# - According to the former, the relative frequency will converge to the true probability. However, a Bayesian application is based on the subjective interpretation. 
# - Unlike the Frequentists, Bayesian statisticians consider the probability distribution as uncertain, and **it is revised as new information comes in**.
# 
# </font>
# </div>

# <div class="alert alert-info">
# <font color=black>
# 
# - Bayes’ theorem is attractive but it comes with a cost, which is analytical intractability and hard to solve analytically. However, there are methods used to approximate this computational issues: 
#     - Quadrature approximation
#     - Maximum a posteriori estimation (MAP)
#     - Grid approach
#     - Sampling-based approach
#     - Metropolis–Hastings
#     - Gibbs sampler
#     - No U-Turn sampler
#     
# </font>
# </div>

# <div class="alert alert-info">
# <font color=black>
# 
# - Both Metropolis–Hastings and Gibbs sampler rests on the Markov chain Monte Carlo (MCMC) method.
# - The Markov chain is a model used to describe the transition probabilities among states. A chain is called Markovian if the probability of the current state st depends only on the most recent state.
# - In a nutshell, the MCMC method helps us gather IID samples from posterior density so that we can calculate the poste‐ rior probability.
#     
# </font>
# </div>

# In[2]:


P = [[0.5, 0.2, 0.3],
     [0.2, 0.3, 0.5],
     [0.2, 0.2, 0.6]]

mc = qe.MarkovChain(P, ('studying', 'travelling', 'sleeping'))
mc.is_irreducible


# In[3]:


states = ['studying', 'travelling', 'sleeping']
initial_probs = [0.5, 0.3, 0.6]
state_space = pd.Series(initial_probs, index=states, name='states')


# In[4]:


q_df = pd.DataFrame(columns=states, index=states)
q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0.5, 0.2, 0.3]
q_df.loc[states[1]] = [0.2, 0.3, 0.5]
q_df.loc[states[2]] = [0.2, 0.2, 0.6]


# In[5]:


def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges
edges_wts = _get_markov_edges(q_df)
pprint(edges_wts)


# In[6]:


G = nx.MultiDiGraph()
G.add_nodes_from(states)
for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)
edge_labels = {(n1, n2):d['label'] for n1, n2, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'mc_states.dot')


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/abdullahkarasan/mlfrm/blob/main/codes/chp_4.ipynb
# - Machine Learning for Financial Risk Management with Python Abdullah Karasan
# 
# </font>
# </div>

# In[ ]:




