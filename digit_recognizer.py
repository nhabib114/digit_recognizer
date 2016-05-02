
# coding: utf-8

# In[19]:

import numpy as np
import pandas as pd
import csv as csv
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import datasets, svm, metrics


# In[20]:

in_file = 'train.csv'
digits = pd.read_csv(in_file)


# In[21]:

display(digits.head())


# In[22]:

# Extract feature (X) and target (y) columns
feature_cols = list(digits.columns[1:])  # all columns but first are features
target_col = digits.columns[0]  # first column is the target/label
#print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_train = digits[feature_cols]  # feature values for all students
y_train = digits[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_train.head()  # print the first 5 rows


# In[23]:

in_file = 'test.csv'
test_digits = pd.read_csv(in_file)
feature_cols = list(test_digits.columns[0:])
X_test = test_digits[feature_cols]

print X_test.head()  # print the first 5 rows


# In[24]:

samparr = test_digits[0:1].as_matrix()
samparr.shape = (28,28)
plt.imshow(samparr, cmap = "Greys")


# In[25]:

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

pca = PCA(n_components=100)
X_transformed = pca.fit_transform(X_train)


# In[26]:

clf = GaussianNB()
clf.fit(X_transformed, y_train)


# In[27]:

X_test_transformed = pca.transform(X_test)


# In[28]:

pred = clf.predict(X_test_transformed)


# In[29]:

prediction_file = open('pred.csv', 'wb')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["ImageId", "Label"])


# In[30]:

for number, i in enumerate(pred):
    prediction_file_object.writerow([number + 1, i])


# In[31]:

prediction_file.close()


# In[ ]:



