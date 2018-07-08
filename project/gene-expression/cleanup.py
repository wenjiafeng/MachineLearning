import itertools
#import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler

with open('/student/fengw/ML/project/gene-expression/data_set_ALL_AML_independent.csv') as testfile:
	test = pd.read_csv(testfile)

with open('/student/fengw/ML/project/gene-expression/data_set_ALL_AML_train.csv') as trainfile:
	train = pd.read_csv(trainfile)

with open('/student/fengw/ML/project/gene-expression/actual.csv') as cancer:
	cancertype = pd.read_csv(cancer)


##remove call columns from table
trainkeep = []
for i in train.columns:
	if "call" not in i:
		trainkeep.append(i)

train = train[trainkeep]

testkeep = []
for i in test.columns:
	if "call" not in i:
		testkeep.append(i)

test = test[testkeep]


train = train.T
test = test.T

##cleanup the "gene description","gene accession number"
train.columns = train.iloc[1]
train = train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

test.columns = test.iloc[1]
test = test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

##combine the label with datasets

train = train.reset_index(drop=True)

##The first 38 rows are labels for training data 
settrain = cancertype[cancertype.patient <= 38].reset_index(drop=True)

train = pd.concat([settrain,train], axis=1)

## 38-72 rows are labels for test data

test = test.reset_index(drop=True)
settest = cancertype[cancertype.patient > 38].reset_index(drop=True)

test = pd.concat([settest,test], axis=1)



#samples
sample = train.iloc[:,2:].sample(n=100, axis=1)
sample["cancer"] = train.cancer
sample.to_csv('/student/fengw/ML/project/gene-expression/sample.csv')

#print sample.describe().round()

#scale the data
scaler = StandardScaler().fit(train.iloc[:,2:])
scaled_train = scaler.transform(train.iloc[:,2:])
scaled_test = scaler.transform(test.iloc[:,2:])

x_train = train.iloc[:,2:]
y_train = train.iloc[:,1]
x_test = test.iloc[:,2:]
y_test = test.iloc[:,1]


#x_train.to_csv('/student/fengw/ML/project/gene-expression/xtrain.csv')
#y_train.to_csv('/student/fengw/ML/project/gene-expression/ytrain.csv')
#x_test.to_csv('/student/fengw/ML/project/gene-expression/xtest.csv')
#y_test.to_csv('/student/fengw/ML/project/gene-expression/ytest.csv')




