#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#X,y=np.arange(10), range(10)
X,y=np.arange(20).reshape((10,2)), range(10)


# In[3]:


X


# In[4]:


y


# In[5]:


list(y)


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[8]:


X_train


# In[9]:


X_test


# In[10]:


y_train


# In[11]:


y_test


# In[12]:


y_train


# In[ ]:





# In[13]:


import pandas as pd


# In[14]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df=pd.read_csv("insurance_data.csv")


# In[16]:


df


# In[17]:


plt.xlabel('age')
plt.ylabel('bought_insurance')
plt.scatter(df.age,df.bought_insurance,color='red',marker='+')


# In[18]:


X=df.drop('bought_insurance',axis='columns')
X


# In[19]:


Y=df.bought_insurance
Y


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[22]:


X_train.shape


# In[23]:


X_train


# In[24]:


X_test.shape


# In[25]:


X_test


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


reg=LinearRegression()


# In[30]:


reg.fit(X,Y);


# In[31]:


reg.coef_


# In[32]:


reg.intercept_


# In[33]:


prediction=reg.predict(X)


# In[34]:


prediction


# In[35]:


plt.xlabel('age')
plt.ylabel('bought_insurance')
plt.scatter(df.age,df.bought_insurance,color='red',marker='+')
plt.plot(X,prediction,'b')


# In[36]:


from sklearn.linear_model import LogisticRegression


# In[37]:


model=LogisticRegression()


# In[39]:


model.fit(X_train,y_train);


# In[40]:


y_predicted=model.predict(X_test)


# In[41]:


y_predicted


# In[42]:


df_new=pd.DataFrame(y_test)


# In[43]:


df_new


# In[44]:


df_new['Predicted']=y_predicted


# In[45]:


df_new


# In[46]:


df_new.columns=['Actuals','Predicted']
df_new


# In[47]:


y_predicted=model.predict([[35]])


# In[48]:


y_predicted


# In[55]:


y_predicted=model.predict([[38]])
y_predicted


# In[57]:


y_predicted=model.predict(X_test)


# In[56]:


from sklearn.metrics import confusion_matrix, classification_report


# In[58]:


print(confusion_matrix(y_test,y_predicted))


# In[59]:


print(classification_report(y_test,y_predicted))


# In[60]:


df=pd.read_csv("Social_Network_Ads.csv")


# In[61]:


df


# In[62]:


df.shape


# In[ ]:




