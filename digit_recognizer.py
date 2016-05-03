import numpy as np
import pandas as pd
import csv as csv
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import datasets, svm, metrics

in_file = 'train.csv'
digits = pd.read_csv(in_file)
display(digits.head())

n_samples = digits.shape[0] # 42000 images
digits_train = digits[:10000] # train on first 10000 values

digits_test = digits[n_samples / 2:]    # test on last half of dataset
display(digits_test.head())

# Extract feature (X) and target (y) columns
feature_cols = list(digits_train.columns[1:])  # all columns but first are features
target_col = digits_train.columns[0]  # first column is the target/label
#print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_train = digits_train[feature_cols]  # feature values 
y_train = digits_train[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_train.head()

# Extract feature (X) and target (y) columns
feature_cols = list(digits_test.columns[1:])  # all columns but first are features
target_col = digits_test.columns[0]  # first column is the target/label
#print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_test = digits_test[feature_cols]  # feature values 
y_test = digits_test[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_test.head() 

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

pca = PCA(n_components=100)     # reduce features with PCA down to 100
X_transformed = pca.fit_transform(X_train) # fit and transform X_train

clf = SVC(kernel='poly', degree=2)
clf.fit(X_transformed, y_train)

X_test_transformed = pca.transform(X_test)  # transform X_test

pred = clf.predict(X_test_transformed)
acc = accuracy_score(y_test, pred)      #accuracy ~ 96%
print acc



