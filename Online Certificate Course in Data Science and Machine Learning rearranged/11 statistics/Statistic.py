#!/usr/bin/env python
# coding: utf-8

# Probability=Desired outcome/total outcome

# Coin Tossing: 
# 
# Two option Head and Tail
# 
# Total Event =2
# 
# P(H)=1/2=0.5
# 
# P(T)=1/2=0.5

# In[1]:


#N_flip=int(input("How many times you are going to toss the conin:"))
N_flip=5


# In[2]:


import numpy as np


# In[3]:


result=np.random.randint(0,2,N_flip)


# In[4]:


print(result)


# In[5]:


index=0
H_cnt=0
T_cnt=0
for i in result:
    if result[index]==0:
        #print("Head")
        H_cnt+=1
    else:
        #print("Tail")
        T_cnt+=1
    index=index+1
print("Head Count:",H_cnt)
print("Tail Count:",T_cnt)


# In[6]:


P_H=H_cnt/N_flip
P_H


# In[7]:


P_T=T_cnt/N_flip
P_T


# In[ ]:





# In[8]:


import pandas as pd

data = {'Actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        }


# In[9]:


df=pd.DataFrame(data, columns=['Actual','Predicted'])


# In[10]:


df.head()


# In[11]:


print(df)


# In[12]:


#TP, TN, FP, FN


# In[13]:


from sklearn.metrics import classification_report, confusion_matrix


# In[14]:


print(confusion_matrix(df.Actual,df.Predicted)) # 0-yes case, 1- No case


# In[19]:


#print(confusion_matrix(df.Predicted,df.Actual))


# In[18]:


print(classification_report(df.Actual,df.Predicted))


# In[ ]:





# In[20]:


#Normal Distribution


# In[21]:


from numpy import random


# In[22]:


x=random.normal(size=10)


# In[23]:


x


# In[24]:


x=random.normal(size=(2,3))


# In[25]:


x


# In[28]:


x=random.normal(loc=1,scale=1,size=(2,3))


# In[29]:


x


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


sns.distplot(random.normal(loc=1,scale=0.2,size=1000), hist=False)


# In[36]:


sns.distplot(random.normal(loc=0,scale=1,size=1000), hist=False)


# In[34]:


x=random.normal(loc=1,scale=1,size=1000)


# In[33]:


x


# In[42]:


#Entropy


# In[43]:


from scipy.stats import entropy


# In[44]:


#p=[1/6,1/6,1/6,1/6, 1/6, 1/6]# case of die


# In[55]:


p=[9/14,5/14]# coin


# In[56]:


e=entropy(p,base=2)


# In[57]:


e


# In[ ]:


#Central lin

