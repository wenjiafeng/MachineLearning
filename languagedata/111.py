#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re  
import csv  
import os



a = open('/student/fengw/ML/languagedata/sorttrain.txt')
output = open('/student/fengw/ML/languagedata/normalize.txt','a')##training data normalization file
line = a.readlines()

for i in line:
    b = str(i) 
    b = b.replace("ö","o")
    b = b.replace("ç","c")
    b = b.replace("ü","u")
    b = b.replace("ə","e")
    b = b.replace("ş","s")
    b = b.replace("ğ","g")
    b = b.replace("ı","i")
    b = b.replace("№","N")
    
    output.write(b)
 
##real test data output
with open('/student/fengw/ML/languagedata/input.csv','r') as datain:
     rows=datain.readlines()
#     with open('/student/fengw/ML/languagedata/test.txt','a') as test:
     with open('/student/fengw/ML/languagedata/test2.txt','a') as test:
          count=0
          for row in rows:
#              data=row.split(',')
               data=row.split(',',1)
#              if data[1]=='"':
#                 test.write(",\n")
#              else:
#                 s=data[1].strip('"')
#                 d=s.strip('\n')
#                 test.write('\"'+d+'\"\n')
               test.write(data[1]) 
               count+=1
print count

output.close()
a.close()


