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
  print("==========================\n")



# use LogisticRegression
# https://www.w3schools.com/python/python_ml_logistic_regression.asp 
from sklearn.linear_model import LogisticRegression

LRG = LogisticRegression(max_iter=200000)
LRG.fit(X_train_scaled, y_train)
LRG_predicted = LRG.predict(X_test_scaled)
report(LRG_predicted, "LRG")



# use RandomForestClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
RFC = RandomForestClassifier() # max_depth=2, 
RFC.fit(X_train_scaled, y_train)
RFC_predict = RFC.predict(X_test_scaled)
report(RFC_predict, "RFC")



# use GuassianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html?highlight=gaussiannb
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train.values, y_train)
GNB_predict = GNB.predict(X_test.values)
report(GNB_predict, "GNB")



# use DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(X_train_scaled, y_train)
DTC_predict = DTC.predict(X_test_scaled)
report(DTC_predict, "DTC")



# use KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier
from sklearn.neighbors import KNeighborsClassifier
KNC = KNeighborsClassifier(n_neighbors=10)
KNC.fit(X_train_scaled, y_train)
KNC_predict = KNC.predict(X_test_scaled)
report(KNC_predict, "KNC")


