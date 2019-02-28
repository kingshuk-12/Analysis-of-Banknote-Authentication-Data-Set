
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:/Users/Kingshuk/Desktop/dataset.csv')
X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


# In[11]:


from sklearn import svm

clf = svm.SVC(kernel='linear') 

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


# In[12]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

