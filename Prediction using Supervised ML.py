#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv("student_scores.csv")
data.head()


# In[5]:


data.plot(x="Hours", y="Scores" , style='o')
plt.title("Hours vs Percentage")
plt.xlabel("The Hours Studied")
plt.ylabel("The Percentage Score")
plt.show()


# In[16]:


X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[19]:


from sklearn.model_selection import train_test_split  
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[21]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)


# In[24]:


hours=[[9.25]]
pred=reg.predict(hours)
pred


# In[ ]:




