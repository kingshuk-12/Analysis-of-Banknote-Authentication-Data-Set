
# coding: utf-8

# In[14]:


import pandas as pd

dataset = pd.read_csv("C:/Users/Kingshuk/Desktop/dataset.csv")
features = dataset.iloc[:, 0:4]
label = dataset.iloc[:, 4]


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[25]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

