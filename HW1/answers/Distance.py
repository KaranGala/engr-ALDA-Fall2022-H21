#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import heapq
from scipy.spatial import distance

sns.set(rc={'figure.figsize':(12,10)})
sns.set_style("whitegrid")


# ## Data sets sometimes must be cleaned before they can be used. For this question, we have left the data set as it was stored online. Parse and clean the data file into a accessible representation e.g. a Pandas DataFrame. You may consider beginning by examining the provided file for possible problems when reading it in.

# In[3]:


df = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', skiprows=[0,124,125,126])
df


# In[5]:


df.dropna(inplace = True)


# In[6]:


df['Classes  '] = df['Classes  '].str.lstrip()
df['Classes  '] = df['Classes  '].str.rstrip()


# In[7]:


df['Classes  '].unique()


# In[8]:


df.info()


# ## Generate a scatter plot between the relative humidity and the wind speed of the observations. Label the axes (relative humidity should be x-axis and wind speed should be y-axis). Call this plot “relative humidity and wind speed”. What general interpretation can you make from this plot?

# In[10]:


sns.scatterplot(data=df, x=' RH', y=' Ws', hue='Classes  ', style="Classes  ")
plt.title('Relative Humidity and Wind Speed')
# Set x-axis label
plt.xlabel('Relative Humidity')
# Set y-axis label
plt.ylabel('Wind Speed')


# ## From the above scatter plot we can clearly infer that when the Wind Speed is kept between 10-20 and if the relative humidity is kept low, the chances of forest fire is much more and for the same wind speed, as the relative humidity increases the chances for the forest fires decreases significantly.

# # Define a data point called P such that P = (mean(RH), mean(Ws)).

# In[11]:


P = (np.mean(df[' RH']), np.mean(df[' Ws']))
P


# # Compute the distance between P and the 244 data points using the following distance measures: 1) Euclidean distance, 2) Manhattan block metric, 3) Minkowski metric (for power=7), 4) Chebyshev distance, and 5) Cosine distance. List the closest 6 points for each distance.

# ### Euclidean distance

# In[37]:


li = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.euclidean(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i])), (df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
print('Closest 6 points to point P are:')
for i in range(6):
    print(heapq.heappop(li)[1])
    heapq.heapify(li)


# ### Manhattan distance

# In[38]:


# Manhattan distance
li = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.cityblock(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i])), (df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
print('Closest 6 points to point P are:')
for i in range(6):
    print(heapq.heappop(li)[1])
    heapq.heapify(li)


# ### Minkowski distance 

# In[39]:


li = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.minkowski(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i]),7), (df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
print('Closest 6 points to point P are:')
for i in range(6):
    print(heapq.heappop(li)[1])
    heapq.heapify(li)


# ### Chebyshev distance

# In[40]:


li = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.chebyshev(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i])),(df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
print('Closest 6 points to point P are:')
for i in range(6):
    print(heapq.heappop(li)[1])
    heapq.heapify(li)


# ### Cosine distance

# In[41]:


li = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.cosine(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i])),(df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
print('Closest 6 points to point P are:')
for i in range(6):
    print(heapq.heappop(li)[1])
    heapq.heapify(li)


# ## For each distance measure, identify the 20 points from the dataset that are the closest to the point P from (b). (You are allowed to use any package functions to calculate the distances.)
# 
# ## i. Create plots, one for each distance measure. Place P on the plot and mark the 20 closest points. To mark them, you could use different colors or shapes. Make sure the points can be uniquely identified.

# In[18]:


li = []
xy_euc = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.euclidean(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i])), (df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
for i in range(20):
    xy_euc.append(heapq.heappop(li)[1])
    heapq.heapify(li)


# In[19]:


x_euc = []
y_euc = []
for i in xy_euc:
    x_euc.append(i[0])
    y_euc.append(i[1])


# In[20]:


li = []
xy_man = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.cityblock(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i])), (df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
for i in range(20):
    xy_man.append(heapq.heappop(li)[1])
    heapq.heapify(li)


# In[21]:


x_man = []
y_man = []
for i in xy_man:
    x_man.append(i[0])
    y_man.append(i[1])


# In[22]:


li = []
xy_min = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.minkowski(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i]),7), (df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
for i in range(20):
    xy_min.append(heapq.heappop(li)[1])
    heapq.heapify(li)


# In[23]:


x_min = []
y_min = []
for i in xy_min:
    x_min.append(i[0])
    y_min.append(i[1])


# In[24]:


li = []
xy_cheb = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.chebyshev(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i])), (df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
for i in range(20):
    xy_cheb.append(heapq.heappop(li)[1])
    heapq.heapify(li)


# In[25]:


x_cheb = []
y_cheb = []
for i in xy_cheb:
    x_cheb.append(i[0])
    y_cheb.append(i[1])


# In[26]:


li = []
xy_cos = []
heapq.heapify(li)
for i in range(len(df)):
    heapq.heappush(li, (distance.cosine(P,(df[' RH'].iloc[i],df[' Ws'].iloc[i])), (df[' RH'].iloc[i],df[' Ws'].iloc[i]),i))
for i in range(20):
    xy_cos.append(heapq.heappop(li)[1])
    heapq.heapify(li)


# In[27]:


x_cos = []
y_cos = []
for i in xy_cos:
    x_cos.append(i[0])
    y_cos.append(i[1])


# In[29]:


colors=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']


# In[31]:


plt.scatter(x_euc, y_euc, c=colors)
plt.title('Euclidean distance')
plt.xlabel('Relative Humidity')
plt.ylabel('Wind Speed')
plt.plot(P[0],P[1],marker="o",markersize=10,markerfacecolor="black")


# In[32]:


plt.scatter(x_man, y_man, c=colors)
plt.title('Manhattan distance')
plt.xlabel('Relative Humidity')
plt.ylabel('Wind Speed')
plt.plot(P[0],P[1],marker="o",markersize=10,markerfacecolor="black")


# In[33]:


plt.scatter(x_min, y_min, c=colors)
plt.title('Minkowski distance')
plt.xlabel('Relative Humidity')
plt.ylabel('Wind Speed')
plt.plot(P[0],P[1],marker="o",markersize=10,markerfacecolor="black")


# In[34]:


plt.scatter(x_cheb, y_cheb, c=colors)
plt.title('Chebyshev distance')
plt.xlabel('Relative Humidity')
plt.ylabel('Wind Speed')
plt.plot(P[0],P[1],marker="o",markersize=10,markerfacecolor="black")


# In[35]:


plt.scatter(x_cos, y_cos, c=colors)
plt.title('Cosine distance')
plt.xlabel('Relative Humidity')
plt.ylabel('Wind Speed')
plt.plot(P[0],P[1],marker="o",markersize=10,markerfacecolor="black")


# ## ii. Verify if the set of points is the same across all the distance measures. If there is any big difference, briefly explain why it is.

# ### The graph shows that the 20 closest points to the mean P are almost similar for all distances other than the cosine distance. Because cosine distance uses the angle between the two vectors created by joining origin to the data point and origin to the mean p, respectively, all other distances use the difference between the coordinates to determine distance, which is why there is a difference between them. As a result, it is clear that the points with the cosine distance that are closest to the mean P are those that make a comparable angle with the vector generated by the mean p and origin.
