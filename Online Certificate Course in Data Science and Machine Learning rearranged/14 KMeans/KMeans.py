#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('income.csv')


# In[5]:


df


# In[6]:


df.shape


# In[7]:


plt.scatter(df['Age'],df['Income'])
plt.xlabel('Age')
plt.ylabel('Income')


# In[8]:


from sklearn.cluster import KMeans


# In[9]:


km=KMeans(n_clusters=3)


# In[10]:


X=df.drop('Name',axis='columns')
X


# In[11]:


km.fit(X);


# In[12]:


y=km.fit_predict(X);
y


# In[13]:


km.cluster_centers_


# In[14]:


df['cluster']=y


# In[15]:


df


# In[16]:


df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]


# In[17]:


plt.scatter(df0['Age'],df0['Income'],color='green')
plt.scatter(df1['Age'],df1['Income'],color='red')
plt.scatter(df2['Age'],df2['Income'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], color='purple',marker='*',label='centroid')


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[24]:


scaler=MinMaxScaler()
df['Income']=scaler.fit_transform(df[['Income']])
df['Age']=scaler.fit_transform(df[['Age']])


# In[25]:


df


# In[27]:


X=df.values[:,1:3]


# In[28]:


X


# In[29]:


km=KMeans(n_clusters=3)


# In[30]:


y_predict=km.fit_predict(X)


# In[31]:


y_predict


# In[32]:


df['cluster']=y_predict
df


# In[33]:


km.cluster_centers_


# In[34]:


df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
plt.scatter(df0['Age'],df0['Income'],color='green')
plt.scatter(df1['Age'],df1['Income'],color='red')
plt.scatter(df2['Age'],df2['Income'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], color='purple',marker='*',label='centroid')


# In[ ]:




