#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


df = pd.read_csv(r"C:\Users\KUNAL SINGH\OneDrive\Desktop\YOMA BYLD TECHNOLOGIES\MACHINE LEARING-DEEP LEARNING\UNSUPERVISED LEARNING\data\income.csv")


# In[33]:


df


# In[34]:


plt.scatter(df.Age,df.Income)
plt.ylabel("Income $")
plt.xlabel("Age")


# In[35]:


km = KMeans(n_clusters = 3)
y_predict = km.fit_predict(df[['Age','Income']])
y_predict


# In[39]:


df['clusters'] = y_predict
df


# In[40]:


km.cluster_centers_


# In[41]:


df1= df[df.clusters==0]
df2= df[df.clusters==1]
df3= df[df.clusters==2]


# In[47]:


plt.scatter(df1.Age,df1.Income,color ='green')
plt.scatter(df2.Age,df2.Income,color ='red')
plt.scatter(df3.Age,df3.Income,color ='black')
plt.ylabel("Income $",rotation = 0)
plt.xlabel("Age")
plt.legend()


# In[82]:


scaler = MinMaxScaler()
scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])

scaler.fit(df[['Age']])
df['Age']= scaler.transform(df[['Age']])
               


x= df.drop('Income_2', axis = 1)
x


# In[87]:


km = KMeans(n_clusters=3)
y_predict = km.fit_predict(df[['Age','Income']])
y_predict


# In[86]:


x['clusters']=y_predict
x


# In[89]:


km.cluster_centers_


# In[91]:


df1= x[x.clusters==0]
df2= x[x.clusters==1]
df3= x[x.clusters==2]


# In[95]:


plt.scatter(df1.Age,df1.Income,color = 'green')
plt.scatter(df2.Age,df2.Income,color = 'red')
plt.scatter(df3.Age,df3.Income,color = 'black') 
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'purple', marker = '*')


# In[104]:


#using sum of square error method to get the error with inertia

sse = []
k_range = range(1,10)
for k in k_range:
    km = KMeans(n_clusters = k)
    km.fit(df[['Age','Income']])
    sse.append(km.inertia_)


# In[105]:


sse


# In[106]:


plt.xlabel('K')
plt.ylabel('sum of square error')
plt.plot(k_range,sse)


# In[ ]:


k = 3 is the most appropiate value to start rate 

