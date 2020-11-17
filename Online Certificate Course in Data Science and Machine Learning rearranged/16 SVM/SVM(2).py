#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.datasets import load_breast_cancer


# In[4]:


cancer=load_breast_cancer()


# In[5]:


cancer


# In[6]:


cancer.keys()


# In[7]:


print(cancer.DESCR)


# In[8]:


cancer.data


# In[9]:


cancer.data.shape


# In[10]:


cancer.data[0]


# In[11]:


cancer.target


# In[12]:


cancer.target_names


# In[13]:


cancer.feature_names


# In[14]:


X=df=pd.DataFrame(cancer.data,columns=cancer.feature_names)


# In[15]:


df


# In[16]:


Y=pd.DataFrame(cancer['target'],columns=['Cancer'])
Y


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[19]:


X_train.shape


# In[20]:


from sklearn.svm import SVC


# In[21]:


model=SVC()


# In[22]:


model.fit(X_train,y_train);


# In[23]:


predictions=model.predict(X_test)


# In[24]:


predictions


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix


# In[26]:


print(classification_report(y_test,predictions))


# In[27]:


model=SVC(C=0.1,gamma=1,kernel='rbf')
model.fit(X_train,y_train);
predictions=model.predict(X_test)
print(classification_report(y_test,predictions))


# In[ ]:




