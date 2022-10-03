#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:


dir(iris)


# In[4]:


iris.feature_names


# In[6]:


iris.target_names


# In[8]:


df = pd.DataFrame(iris.data , columns = iris.feature_names)
df.head()


# In[10]:


df['target'] = iris.target
df.head()


# In[11]:


df.shape


# In[32]:


df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]


# In[33]:


df0.head()


# In[34]:


df1.head()


# In[35]:


df2.head()


# In[36]:


df['flower_names'] = df.target.apply(lambda x : iris.target_names[x])
df


# In[37]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color = 'green')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color = 'red')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color = 'blue')


# In[42]:


plt.xlabel('petal length')
plt.ylabel('petal width')
plt.scatter(df0['petal length (cm)'],df0['petal length (cm)'],color = 'green')
plt.scatter(df1['petal length (cm)'],df1['petal length (cm)'],color = 'red')
plt.scatter(df2['petal length (cm)'],df2['petal length (cm)'],color = 'blue')


# In[43]:


df.head()


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


#train split

x = df.drop(['target','flower_names'],axis = 'columns')
y = df.target


# In[49]:


x_train, x_test , y_train, y_test = train_test_split(x,y, test_size = 0.2)


# In[50]:


len(x_train)


# In[51]:


len(x_test)


# In[52]:


len(y_train)


# In[70]:


len(y_test)


# # creating KNN classifier

# In[100]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)


# In[101]:


knn.fit(x_train,y_train)


# In[102]:


knn.score(x_test,y_test)


# In[106]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
cm


# In[118]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot = True)
plt.xlabel("predicted")
plt.ylabel('truth')


# In[126]:


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))


# In[ ]:




