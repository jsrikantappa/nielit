#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv('Decision_Tree_ Dataset -1.csv')


# In[4]:


data.head()


# In[5]:


data.info


# In[6]:


X=data.values[:,0:5]
X


# In[7]:


Y=data.values[:,5]
Y


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[10]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[12]:


clf=DecisionTreeClassifier(criterion='entropy')


# In[13]:


clf.fit(X_train,y_train)


# In[14]:


y_predict=clf.predict(X_test)
y_predict


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix


# In[16]:


print(classification_report(y_test,y_predict))


# In[17]:


print(confusion_matrix(y_test,y_predict))


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


cm=confusion_matrix(y_test,y_predict)


# In[17]:


cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True,fmt=".0f")
plt.xlabel('Predict')
plt.ylabel('Actual')


# In[23]:


#Random Forest


# In[1]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


digit=load_digits()


# In[3]:


digit


# In[4]:


digit.keys()


# In[29]:


print(digit.DESCR)


# In[31]:


digit.keys()


# In[33]:


digit.data.shape


# In[34]:


digit.data[0]


# In[35]:


digit.target


# In[36]:


digit.target.shape


# In[38]:


digit.target[0:50]


# In[39]:


digit.target_names


# In[41]:


digit.images.shape


# In[42]:


digit.images[0]


# In[43]:


import pylab as pl
pl.gray()
pl.matshow(digit.images[0])


# In[44]:


for i in range(5):
    pl.matshow(digit.images[i]);


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(digit.data, digit.target, test_size=0.3, random_state=42)


# In[7]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[9]:


from sklearn.tree import DecisionTreeClassifier


# In[10]:


clf=DecisionTreeClassifier()


# In[11]:


clf.fit(X_train,y_train);


# In[12]:


y_predict=clf.predict(X_test)


# In[13]:


y_predict


# In[14]:


from sklearn.metrics import classification_report, confusion_matrix


# In[15]:


print(classification_report(y_test,y_predict))


# In[16]:


print(confusion_matrix(y_test,y_predict))


# In[18]:


cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True,fmt=".0f")
plt.xlabel('Predict')
plt.ylabel('Actual')


# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


rfc=RandomForestClassifier(n_estimators=600)


# In[21]:


rfc.fit(X_train,y_train);


# In[22]:


prediction=rfc.predict(X_test)


# In[23]:


print(classification_report(y_test,prediction))


# In[24]:


cm=confusion_matrix(y_test,prediction)
sns.heatmap(cm,annot=True,fmt=".0f")
plt.xlabel('Predict')
plt.ylabel('Actual')


# In[ ]:




