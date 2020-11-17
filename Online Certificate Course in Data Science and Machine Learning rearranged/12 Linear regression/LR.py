#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df=pd.read_csv('homeprices.csv')


# In[10]:


df


# In[11]:


df.shape


# In[13]:


plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red')


# In[14]:


X=df.drop('price',axis='columns')
X


# In[15]:


Y=df.price
Y


# In[20]:


#from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[21]:


reg=LinearRegression()


# In[22]:


reg.fit(X,Y);


# In[24]:


reg.coef_       #m


# In[25]:


reg.intercept_    #c


# In[27]:


reg.predict([[3300]])


# In[28]:


#y = mx+c


# In[29]:


(reg.coef_)*3300+reg.intercept_ 


# In[30]:


reg.predict([[5000]])


# In[31]:


predictions=reg.predict(X)
predictions


# In[32]:


df


# In[34]:


plt.plot(X,Y,'r') #actual
plt.plot(X,predictions,'b') # predicted


# In[35]:


from sklearn import metrics


# In[36]:


metrics.mean_absolute_error(Y,predictions)


# In[37]:


metrics.mean_squared_error(Y,predictions)


# In[38]:


np.sqrt(metrics.mean_squared_error(Y,predictions))


# In[ ]:




