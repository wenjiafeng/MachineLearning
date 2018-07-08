#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

with open('/student/fengw/ML/project/gene-expression/xtrain.csv') as xtrain:
        x_train=pd.read_csv(xtrain,header=1)
with open('/student/fengw/ML/project/gene-expression/ytrain.csv') as ytrain:
        y_train=pd.read_csv(ytrain,header=0)
with open('/student/fengw/ML/project/gene-expression/xtest.csv') as xtest:
        x_test=pd.read_csv(xtest,header=1)
with open('/student/fengw/ML/project/gene-expression/ytest.csv') as ytest:
        y_test=pd.read_csv(ytest,header=0)

y_train=y_train.replace('ALL',0)
y_train=y_train.replace('AML',1)
y_test=y_test.replace('ALL',0)
y_test=y_test.replace('AML',1)

x_train=x_train.iloc[0:-1,1:-1]
y_train=y_train.iloc[0:-1,1]
x_test=x_test.iloc[0:-1,1:-1]
y_test=y_test.iloc[0:-1,1]

#data scaling
sc=StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train_scale = sc.transform(x_train)
x_test_scale = sc.transform(x_test)


clf = GaussianNB()
clf.fit(x_train,y_train)

A=accuracy_score(clf.predict(x_test),y_test)
pred = clf.predict(x_test)
M=confusion_matrix(y_test,pred)

print 'Accuracy for Naive Bayes using unscaled features: '
print A

print 'Confusion Matrix : '
print M

clf.fit(x_train_scale,y_train)
Acc = accuracy_score(clf.predict(x_test_scale),y_test)
Mat = confusion_matrix(y_test,clf.predict(x_test_scale))

print 'Accuracy for Naive Bayes using scaled features: '
print Acc

print 'Confusion Matrix : '
print Mat






