#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install yfinance')
get_ipython().system('pip install plotly')


# In[6]:


import pandas as pd
import numpy as np
import yfinance as yf
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff


# In[18]:


#-- Random assets : Disney, Microsoft, Amazon, Exxon, Apple, Gold ETF, Eni, Ford, Procter&Gamble, Pfizer
ticker = ['DIS','MSFT',
 'AMZN', 'XOM',
 'AAPL', 'GLD',
 'E', 'F',
 'PG', 'PFE']
pf_data = pd.DataFrame()
for t in ticker:
 pf_data[t]=yf.download(t, start='2005-01-01', end='2023-01-27')['Adj Close']
# check with --> pf_data.info()


# In[19]:


#-- Calculate logarithmic returns
log_ret = np.log(1 + pf_data.pct_change())
#-- Create an object with the number of tickers
num_ass = len(ticker)


# In[20]:


#-- Create empty lists
pf_ret=[]
pf_vol=[]
dis_weight=[]
msft_weight=[]
amzn_weights=[]
xom_weights =[]
aapl_weights=[]
gld_weights=[]
e_weights = []
f_weights = []
pg_weights = []
pfe_weights = []


# In[22]:


#-- Create a loop with 'n' (e.g 10,000) iteractions that will generate random weights[sum(Wi)=1],
#-- and will append returns and volatilities to the (empty) lists specified above
for x in range(10000):
 weights=np.random.random(num_ass)
 weights /= np.sum(weights)
 pf_ret.append(np.sum(weights*log_ret.mean()*250))
 pf_vol.append(np.sqrt(np.dot(weights.T,
 np.dot(log_ret.cov()*250,
 weights))))
#-- Append the weight values for all tickers
 dis_weight.append(weights[0])
 msft_weight.append(weights[1])
 amzn_weights.append(weights[2])
 xom_weights.append(weights[3])
 aapl_weights.append(weights[4])
 gld_weights.append(weights[5])
 e_weights.append(weights[6])
 f_weights.append(weights[7])
 pg_weights.append(weights[8])
 pfe_weights.append(weights[9])


# In[23]:


#-- Transform the lists into NumPy arrays
pf_ret = np.array(pf_ret)
pf_vol = np.array(pf_vol)
dis_weight = np.array(dis_weight)
msft_weight = np.array(msft_weight)
amzn_weights = np.array(amzn_weights)
xom_weights= np.array(xom_weights)
aapl_weights = np.array(aapl_weights)
gld_weights = np.array(gld_weights)
e_weights = np.array(e_weights)
f_weights = np.array(f_weights)
pg_weights = np.array(pg_weights)
pfe_weights = np.array(pfe_weights)


# In[24]:


#-- Create a dictionary containing all the portfolios obtained previously
portfolios = pd.DataFrame(
 {'Return': pf_ret,
 'Volatility' : pf_vol,
 'Disney Weight' : dis_weight,
 'Microsoft Weight' : msft_weight,
 'Amazon Weight': amzn_weights ,
 'Exxon Weight' : xom_weights,
 'Apple Weight' : aapl_weights,
 'Gold Weight' : gld_weights,
 'Eni Weight' : e_weights,
 'Ford Weight' : f_weights,
 'P&G Weight' : pg_weights,
 'Pfizer Weight' : pfe_weights
 }
 )
portfolios


# In[25]:


#-- Include risk free rate
#-- Assume a risk free rate equal to 2.5%
risk_free_rate=0.025
fig = go.Figure()
fig.add_trace(go.Scatter(x=pf_vol, y=pf_ret,
 marker=dict(color=(pf_ret-risk_free_rate)/pf_vol,#Sharpe Ratio
 showscale=True,
 size=7,
line=dict(width=1),
 colorscale="rdbu",
colorbar=dict(title="Sharpe<br>Ratio")
 ),
 mode='markers'))
fig.update_layout(template='plotly_white',
 xaxis=dict(title='Volatility'),
 yaxis=dict(title='Return'),
 title='Dynamic Scatterplot --- 10,000 Random Portfolios',
 width=850,
 height=500)
fig.update_xaxes(range=[0.125, 0.27]) #-- fix it as you prefer
fig.update_yaxes(range=[0.03,0.185]) #-- fix it as you prefer
fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
#-- As output you will obtain a dynamic graph, where you can either zoom in and out,
#-- see each portfolio's risk&return, or crop the graph as you want.


# In[27]:


#-- Choose the optimized portfolio with a given range of Volatility
#-- (e.g. between 17% and 18%)
pf1_maxret = portfolios[(portfolios['Volatility']>=.17)&(portfolios['Volatility']<=.18)].max()['Return']
np.where(portfolios['Return']== pf1_maxret)
portfolios.iloc[np.where(portfolios['Return']== pf1_maxret)]


# In[28]:


#-- Choose the optimized portfolio with less Volatility
pf2_minvol = portfolios.min()['Volatility']
portfolios.iloc[np.where(portfolios['Volatility']== pf2_minvol)]


# In[ ]:




