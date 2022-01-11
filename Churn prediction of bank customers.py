#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
import datetime as dt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab
pd.set_option('display.max_columns', None)


# In[6]:


dataset = pd.read_csv('./Downloads/Churn_Modelling.csv')


# In[7]:


print(dataset.shape)
dataset.head()


# In[8]:


dataset= dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis =1)


# In[9]:


dataset.Exited.value_counts()


# In[10]:


cat_var = [col for col in dataset.columns if dataset[col].dtype =='O']
cat_var


# In[12]:


for col in ['Geography', 'Gender']:
    print(col,':')
    print( dataset[col].value_counts())
    print()


# In[13]:


sns.pairplot(dataset, hue='Exited')


# In[14]:


plt.figure(figsize =(22,12))
sns.heatmap(dataset.corr(), annot= True, cmap ='RdYlGn')


# In[16]:


dataset.isnull().sum()


# In[18]:


import scipy.stats as stats


# In[19]:


for col in ['CreditScore','Age','Balance','EstimatedSalary' ]:
    plt.figure(figsize=(14,8))
    plt.subplot(1,3,1)
    dataset[col].plot.hist(bins =30)
    plt.title(col)
    
    plt.subplot(1,3,2)
    stats.probplot(dataset[col], plot=pylab)
    plt.title(col)
    
    plt.subplot(1,3,3)
    dataset[col].plot.box()
    plt.title(col)
    plt.show()


# In[20]:


df =pd.concat([pd.get_dummies(dataset['Gender'], drop_first =True, prefix='Gender'),
              pd.get_dummies(dataset['Geography'], drop_first=True, prefix ='Geography'),
              dataset.drop(['Gender', 'Geography'], axis =1)], axis =1)


# In[21]:


df.head()


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Exited', axis =1), df.Exited, test_size =0.2, random_state =0)
X_train.shape, X_test.shape


# In[23]:


X_train.describe()


# In[24]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled= sc.fit_transform(X_train)
X_test_scalled = sc.transform(X_test)


# In[25]:


from sklearn.metrics import roc_auc_score

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict_proba(X_train)
print(roc_auc_score(y_train, y_pred[:,1]))

y_pred = xgb_model.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:,1]))


# In[26]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred =ada.predict_proba(X_train)
print(roc_auc_score(y_train, y_pred[:,1]))

y_pred = ada.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:,1]))


# In[27]:


from sklearn.ensemble import GradientBoostingClassifier
GBC =GradientBoostingClassifier()

GBC.fit(X_train, y_train)
y_pred =GBC.predict_proba(X_train)
print(roc_auc_score(y_train, y_pred[:,1]))

y_pred = GBC.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:,1]))


# In[28]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict_proba(X_train_scaled)
print(roc_auc_score(y_train, y_pred[:,1]))

y_pred = lr.predict_proba(X_test_scalled)
print(roc_auc_score(y_test, y_pred[:,1]))

