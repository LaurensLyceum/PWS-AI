import pandas as pd
import matplotlib as plt
import numpy as np

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
  
# load database
df=pd.read_csv('MalwareData.csv',sep="|")
# drop 'Name' and 'md5' as we're not using those
df=df.drop(['Name', 'md5'],axis=1)

# set X to all columns other then dropped ones and 'Legitimate' 
X=df.iloc[:,:-1]
# set y to 'Legitimate' column
y=df.iloc[:,-1]

# create train and test sets for X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scale datasets to prevent errors
scaler = preprocessing.StandardScaler().fit(X_train)
scaler

scaler.mean_
scaler.scale_

X_train_scaled = scaler.transform(X_train)
X_train_scaled
X_test_scaled = scaler.transform(X_test)
X_test_scaled

def report(y_predicted, type):
  # print confusion matrix, accuracy and classification report 
  # https://www.w3schools.com/python/python_ml_confusion_matrix.asp
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
  print("========== " + type + "  ==========")
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
logr = linear_model.LogisticRegression(max_iter=2000000000)
logr.fit(X_train_scaled, y_train)
predicted = logr.predict(X_test_scaled)
report(predicted, "logr")