#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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
x_train_scale=sc.transform(x_train)
x_test_scale=sc.transform(x_test)

##decision tree 

#tree_clf = DecisionTreeClassifier(random_state=4)
tree_clf = DecisionTreeClassifier(max_depth=10,random_state=0, criterion = 'entropy')
tree_clf.fit(x_train,y_train)
pred=tree_clf.predict(x_test)
A=accuracy_score(pred,y_test)
M=confusion_matrix(y_test,pred)

print 'Accuracy for DecisionTree using unscaled features: '
print A

print 'Confusion Matrix : '
print M

tree_clf.fit(x_train_scale,y_train)
Acc = accuracy_score(tree_clf.predict(x_test_scale),y_test)
Mat = confusion_matrix(y_test,tree_clf.predict(x_test_scale))

print 'Accuracy for DecisionTree using scaled features: '
print Acc

print 'Confusion Matrix : '
print Mat


##ramdom forest

random_clf = RandomForestClassifier(max_depth=6, random_state=0)
random_clf.fit(x_train,y_train)
pred=tree_clf.predict(x_test)

Ac=accuracy_score(pred,y_test)
Ma=confusion_matrix(y_test,pred)

print 'Accuracy for Random Forest using unscaled features: '
print Ac

print 'Confusion Matrix : '
print Ma

random_clf.fit(x_train_scale,y_train)
Accu = accuracy_score(random_clf.predict(x_test_scale),y_test)
Matr = confusion_matrix(y_test,random_clf.predict(x_test_scale))

print 'Accuracy for Random Forest using scaled features: '
print Accu

print 'Confusion Matrix : '
print Matr




