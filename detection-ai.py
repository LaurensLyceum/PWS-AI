import pandas as pd
import matplotlib as plt
import numpy as np

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
  
# load database
df=pd.read_csv('datasets/MalwareData.csv',sep="|")
# drop 'Name' and 'md5' as we're not using those
df=df.drop(['Name', 'md5'],axis=1)

# set X to all columns other then dropped ones and 'Legitimate' 
X=df.iloc[:,:-1]
# set y to 'Legitimate' column
y=df.iloc[:,-1]

# create training and test sets for X and y
# training set is 70% of the DB
# test set is 30% of the DB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# scale datasets to make it more accurate
# https://scikit-learn.org/stable/modules/preprocessing.html
scaler = preprocessing.StandardScaler().fit(X_train)
scaler

scaler.mean_
scaler.scale_

X_train_scaled = scaler.transform(X_train)
X_train_scaled
X_test_scaled = scaler.transform(X_test)
X_test_scaled

# print report function so I don't have to write it a million times
def report(y_predicted, type):
  # print confusion matrix, accuracy and classification report 
  # https://www.w3schools.com/python/python_ml_confusion_matrix.asp
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
  print("========== " + type + "  ==========")
  # True Negative (Top-Left Quadrant)
  # False Positive (Top-Right Quadrant)
  # False Negative (Bottom-Left Quadrant)
  # True Positive (Bottom-Right Quadrant)
  print("Confusion matrix:")
  print(confusion_matrix(y_test, y_predicted))
  print("\n")
  print("Accuracy:")
  print(accuracy_score(y_test, y_predicted))
  print("\n")
  print("Classification report:")
  print(classification_report(y_test, y_predicted))
  print("============================")


# use logistical regression
# https://www.w3schools.com/python/python_ml_logistic_regression.asp 
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(max_iter=200000)
logr.fit(X_train_scaled, y_train)
logr_predicted = logr.predict(X_test_scaled)
report(logr_predicted, "logr")



# use random forest classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
clf = RandomForestClassifier() # max_depth=2, 
clf.fit(X_train_scaled, y_train)
RFC_predict = clf.predict(X_test_scaled)
report(RFC_predict, "RFC")
