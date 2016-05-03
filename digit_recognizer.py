
# coding: utf-8

# In[103]:

import numpy as np
import pandas as pd
import csv as csv
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#get_ipython().magic(u'matplotlib inline')

from sklearn import datasets, svm, metrics


# In[104]:

in_file = 'train.csv'
digits = pd.read_csv(in_file)


# In[105]:

display(digits.head())


# In[106]:

n_samples = digits.shape[0]
digits_train = digits[:10000]

#in_file = 'test.csv'
#digits_test = pd.read_csv(in_file)

digits_test = digits[n_samples / 2:]

display(digits_test.head())


# In[107]:

# Extract feature (X) and target (y) columns
feature_cols = list(digits_train.columns[1:])  # all columns but first are features
target_col = digits_train.columns[0]  # first column is the target/label
#print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_train = digits_train[feature_cols]  # feature values 
y_train = digits_train[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_train.head() 


# In[108]:

# Extract feature (X) and target (y) columns
feature_cols = list(digits_test.columns[1:])  # all columns but first are features
target_col = digits_test.columns[0]  # first column is the target/label
#print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_test = digits_test[feature_cols]  # feature values 
y_test = digits_test[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_test.head() 


# In[109]:

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

pca = PCA(n_components=100)
X_transformed = pca.fit_transform(X_train)
#X_norm = normalize(X_train)


# In[110]:

clf = SVC(kernel='poly', degree=2)
clf.fit(X_transformed, y_train)
#clf.fit(X_norm, y_train)


# In[111]:

X_test_transformed = pca.transform(X_test)
#X_test_norm = normalize(X_test)


# In[112]:

pred = clf.predict(X_test_transformed)
#pred = clf.predict(X_test_norm)
acc = accuracy_score(y_test, pred)
print acc



