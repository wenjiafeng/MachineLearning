#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import sys
import codecs
import csv
import cStringIO
# read in the file mapping.txt as a python dictionary
# first column are the keys and the second the values
# ignore any word if it's already in the dictionary!!!

# the read the test data and just use the dictionary to
# choose the correct restored word.

# lookup how to open a text file in utf-8 encoding!



## open mapping in utf-8 encoding and creat a dict named 'ref'
#with codecs.open('/student/fengw/ML/languagedata/mapping.txt','r',encoding='utf8') as train:
with codecs.open('/student/fengw/ML/languagedata/mapping.txt','r') as train:
    ref={}

    line=train.readlines()
    for i in line:
        temp = i.split('\t')
        tt=temp[0].replace("Ə","E")
        tt=tt.replace("Ç","C")
        tt=tt.replace("İ","I")
        tt=tt.replace("Ş","S")
        tt=tt.replace("Ö","O")
        tt=tt.replace("Ü","U")
        tt=tt.replace("Ğ","G")
        if tt not in ref:
           ref[tt]=temp[1]
#       ref[temp[0]] = []
#       ref[temp[0]].append(temp[1].strip())
        else:
           pass
#print '111'

## open testfile, find if the element match to ref's key, return to the value if it been found
with open('/student/fengw/ML/languagedata/testclean2.txt','r') as infile:
#with open('/student/fengw/ML/languagedata/cleantest.txt','r') as infile:
     result=[] ## contain correct match with unicode format 
     line1=infile.readlines()
     for j in line1:
         a = j.strip()
         if a in ref:
            result.append(ref[a])
         else:
#            b=unicode(a,"utf-8")
#            result.append(b)
            result.append(a)
#print '222'

##create output file
#tr=codecs.open('/student/fengw/ML/languagedata/output3.txt','w',encoding='utf8')
tr=codecs.open('/student/fengw/ML/languagedata/output3.txt','w')
tr.write('id,token\n')
count=1
for k in result:
    s=k.strip('\n')
    tr.write(str(count)+',\"'+s+'\"\n')
    count+=1

tr.close()



