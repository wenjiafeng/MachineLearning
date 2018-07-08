#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re


a = open ('/student/fengw/ML/languagedata/test2.txt')
clean = open('/student/fengw/ML/languagedata/testclean2.txt','a')
line = a.readlines()

for i in line:
    if i != 1:
       b = i.replace("\"","")
       clean.write(b)

clean.close()
a.close()
