#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[6]:


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


# In[7]:


data.data


# In[8]:


dir(data)


# In[9]:


data.target_names


# In[10]:


data.feature_names


# In[11]:


data.target


# In[15]:


df = pd.DataFrame(np.c_ [data.data,data.target], columns= [list(data.feature_names)+['target']])
df                                                                                     


# In[16]:


df.head()


# In[17]:


df.tail()


# In[19]:


df.shape


# In[21]:


x = df.iloc[:,0:-1]
y = df.iloc[:,-1]


# In[22]:


from sklearn.model_selection import train_test_split
x_train ,x_test, y_train , y_test = train_test_split(x,y,test_size = 0.2)


# In[23]:


x_train.shape


# In[24]:


x_test.shape


# # naive bayes classifier model 

# In[43]:


from sklearn.naive_bayes import GaussianNB
model =  GaussianNB()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[41]:


from sklearn.naive_bayes import MultinomialNB
model_1 =  MultinomialNB()
model_1.fit(x_train,y_train)
model_1.score(x_test,y_test)


# In[42]:


from sklearn.naive_bayes import BernoulliNB
model_2 =  BernoulliNB()
model_2.fit(x_train,y_train)
model_2.score(x_test,y_test)


# In[ ]:





# In[ ]:




