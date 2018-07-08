#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import sys
import codecs
import csv
import cStringIO
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
#import tensorflow

with codecs.open('/student/fengw/ML/languagedata/azj-train.txt','r') as train:
	line = train.readlines()
data=[]
labels=[]
for i in line:
	data.append([i])
#        temp = i.split('\t')
	
        tt=i.replace("Ə","E")
        tt=tt.replace("Ç","C")
        tt=tt.replace("İ","I")
        tt=tt.replace("Ş","S")
        tt=tt.replace("Ö","O")
        tt=tt.replace("Ü","U")
        tt=tt.replace("Ğ","G")
	tt = tt.replace("ö","o")
    	tt = tt.replace("ç","c")
    	tt = tt.replace("ü","u")
    	tt = tt.replace("ə","e")
    	tt = tt.replace("ş","s")
    	tt = tt.replace("ğ","g")
    	tt = tt.replace("ı","i")
    	tt = tt.replace("№","N")
	labels.append(tt)

x_train = np.array(data)
y_train = np.array(labels)


with open('/student/fengw/ML/languagedata/testclean2.txt','r') as infile:
     line1=infile.readlines()
test=[]
for j in line1:
	test.append(j)

x_test = np.array(test)

clf = MLPClassifier()
clf.fit(x_train,y_train)

prediction = clf.predict(x_test) 

result = prediction.tolist() 

print prediction

tr=codecs.open('/student/fengw/ML/languagedata/output3.txt','w')
tr.write('id,token\n')   
count = 1
for k in result:
    s=k.strip('\n')
    tr.write(str(count)+',\"'+s+'\"\n')
    count+=1

tr.close()



