#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('tripadvisor_review.csv',index_col=['User ID'])


# In[3]:


df.head()


# In[4]:


cormat = df.corr()
top_corr_features = cormat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters = 2)


# In[8]:


km.fit(df)


# In[9]:


km.cluster_centers_


# In[10]:


df.shape


# In[11]:


df.isnull().sum()


# In[12]:


df.hist(bins=15,color='steelblue',edgecolor='black',linewidth=1.0,xlabelsize=8,ylabelsize=8,grid=False)
plt.tight_layout(rect=(0, 0, 1.2, 1.2))


# In[13]:


pp=sns.pairplot(df,height=1.8,aspect=1.8,
                plot_kws=dict(edgecolor="k",linewidth=0.5),
                diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig
fig.subplots_adjust(top=0.93,wspace=0.3)


# In[14]:


x=df['Category 7']
y=df['Category 3']
plt.scatter(x,y)

plt.show()


# In[15]:


x=df['Category 6']
y=df['Category 5']
plt.scatter(x,y)
plt.show()


# In[16]:


from sklearn.preprocessing import normalize
data_scales = normalize(df)
data_scales = pd.DataFrame(data_scales,columns=df.columns)
data_scales.head()


# In[17]:


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title("Dendrogram")
dend=shc.dendrogram(shc.linkage(df,method='ward'))


# In[18]:


plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scales, method='ward'))
plt.axhline(y=6, color='r', linestyle='--')


# In[45]:


from sklearn.cluster import AgglomerativeClustering
df1 = df[['Category 3','Category 7']]

cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
cluster.fit_predict(df1)


# In[46]:


plt.figure(figsize=(10,10))
plt.scatter(df['Category 7'],df['Category 3'],c=cluster.labels_,cmap='rainbow')


# In[48]:


plt.figure(figsize=(10,10))
plt.scatter(df['Category 3'],df['Category 7'],c=cluster.labels_,cmap='rainbow')


# In[ ]:




