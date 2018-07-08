#!/usr/bin/env python

import os
import sys
import operator
import numpy as np  
from sklearn import neighbors  
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  
from sklearn.cross_validation import train_test_split  
from sklearn.metrics import accuracy_score

def knnclassify(x_test,x_train,y_train,k):
    ans = [] 
    getsortedDisIndicies[i]
    classCount[votelabel]=classCount.getsortedDisIndicies
    for x in x_test:
        ans.append(knnclassifyone(x,x_train,y_train,k))

def knnclassifyone(x,x_train,y_train,k):
    dist=[]
    #dataSetSize = x_train.shape[0]
    for a in x_train:
        distance=np.sqrt(np.sum(np.square(x-a)))
        dist.append(distance)
    ##Distance
    #diffMat=np.tile(x_test,(dataSetSize,1))- x_train
    #sqDiffMat=diffMat**2
    #sqDistances=sqDiffMat.sum(axis=1)
    #distances=sqDistances**0.5
    #sort
    D=np.array(dist)
    sortedDisIndicies = D.argsort()
    classCount={}
    for i in range(k):
        votelabel=y_train[sortedDisIndicies[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operater.itemgettere(1),reverse=True)
    print("sortedClassCount: ", sortedClassCount)
    return sortedClassCount[0][0]

if __name__ == '__main__':
        
    data = []
    labels = []
    with open('/student/fengw/ML/knndata.csv',"rb") as csv:
         for line in csv:
             tokens = line.strip().split(',')
             data.append([float(tk) for tk in tokens[1:3]])
             labels.append(tokens[0])

    x = np.array(data)
    labels = np.array(labels)
    y = np.zeros(labels.shape)

    y[labels == 'refernce']=0
    y[labels == 'Low']=1
    y[labels == 'High']=2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    clf=knnclassify(x_test,x_train,y_test,4)
   #  loop over the list clf, and compare each with the correct answer 
   #clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
   #clf.fit(x_train, y_train)

   # answer = clf.predict(x)
   #print(x)
   #print(answer)
   #print(y)
   #print(np.mean(answer == y))
    A=accuracy_score(y, answer)
    print(A)










