#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix

import warnings
warnings.filterwarnings('ignore')

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('../Downloads//diabetes.csv')  #import dataset
data.head(10)


# In[5]:


import missingno as msno
msno.bar(data)
plt.show()


# In[6]:


sns.heatmap(data.corr(),cbar=False,cmap='BuGn',annot=True)


# In[7]:


col=['Glucose' ,'BloodPressure' ,'SkinThickness', 'Insulin' ,'BMI']


# In[8]:


for i in col:
  data[i].replace(0,data[i].mean(),inplace=True)


# In[9]:


p=data.hist(figsize = (20,20))


# In[10]:


sns.pairplot(data,hue='Outcome')


# In[11]:


sns.stripplot(x='Pregnancies',y='Age',data=data)


# In[14]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(data.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
y=data.Outcome


# In[15]:


#splitting the dataset
from sklearn.model_selection import train_test_split        
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=.30,random_state=3)


# In[16]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1,penalty='l2')
log_reg.fit(X_train,Y_train)

log_acc=accuracy_score(Y_test,log_reg.predict(X_test))


print("Train Set Accuracy:"+str(accuracy_score(Y_train,log_reg.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,log_reg.predict(X_test))*100))


# In[17]:


# KNearestNeighbors

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)                #knn classifier
knn.fit(X_train,Y_train)

knn_acc = accuracy_score(Y_test,knn.predict(X_test))


print("Train Set Accuracy:"+str(accuracy_score(Y_train,knn.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,knn.predict(X_test))*100))


# In[19]:


# SVC

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train,Y_train)    

svm_acc= accuracy_score(Y_test,svm.predict(X_test))


print("Train Set Accuracy:"+str(accuracy_score(Y_train,svm.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,svm.predict(X_test))*100))


# In[20]:


# DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy',max_depth=5)
dtc.fit(X_train, Y_train)


dtc_acc= accuracy_score(Y_test,dtc.predict(X_test))

print("Train Set Accuracy:"+str(accuracy_score(Y_train,dtc.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,dtc.predict(X_test))*100))


# In[18]:


# GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train)


gbc_acc=accuracy_score(Y_test,gbc.predict(X_test))

print("Train Set Accuracy:"+str(accuracy_score(Y_train,gbc.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,gbc.predict(X_test))*100))


# In[21]:


# XGBClassifier

from xgboost import XGBClassifier

xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth=6,n_estimators = 10)
xgb.fit(X_train,Y_train)

xgb_acc= accuracy_score(Y_test,xgb.predict(X_test))

print("Train Set Accuracy:"+str(accuracy_score(Y_train,xgb.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,xgb.predict(X_test))*100))


# In[22]:


#Stacking

#Stacking is the ensemble technique.In this,two or more classifiers act 
#as base model and the prediction of those will be the x value for the other model(i.e meta-data)

from sklearn.model_selection import train_test_split                #splitting the dataset
                                                                 
train,val_train,test,val_test = train_test_split(X,y,test_size=.50,random_state=3)
x_train,x_test,y_train,y_test = train_test_split(train,test,test_size=.20,random_state=3)


# In[23]:


#first model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)


# In[24]:


# second model
svm = SVC()
svm.fit(x_train, y_train)


# In[25]:


pred_1=knn.predict(val_train)
pred_2=svm.predict(val_train)

# addition of 2 predictions
result = np.column_stack((pred_1,pred_2))


# In[26]:


pred_test1=knn.predict(x_test)
pred_test2=svm.predict(x_test)


predict_test=np.column_stack((pred_test1,pred_test2))


# In[27]:


# stacking classifier
#RandomForestClasifier:- In this prediction of other 2 classification is taken as x value
from sklearn.ensemble import RandomForestClassifier
rand_clf = RandomForestClassifier()
rand_clf.fit(result,val_test) 


# In[28]:


rand_clf.score(result,val_test)


# In[29]:


rand_acc = accuracy_score(y_test ,rand_clf.predict(predict_test))
rand_acc


# In[30]:


models = pd.DataFrame({
    'Model': ['Logistic','KNN', 'SVC',  'Decision Tree Classifier',
             'Gradient Boosting Classifier',  'XgBoost','Stacking'],
    'Score': [ log_acc,knn_acc, svm_acc, dtc_acc, gbc_acc, xgb_acc,rand_acc,]
})

models.sort_values(by = 'Score', ascending = False)


# In[31]:


colors = ["purple", "green", "orange", "magenta","blue","black"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,8))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=models['Model'],y=models['Score'], palette=colors )
plt.show()

