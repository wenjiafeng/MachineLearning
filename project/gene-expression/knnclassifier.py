#!/usr/bin/env python
import os
import sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# RandomizedSearch for tuning (possibly faster than GridSearch)
from sklearn.model_selection import RandomizedSearchCV
# Bayessian optimization supposedly faster than GridSearch
#from bayes_opt import BayesianOptimization

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def cherchez(estimator, param_grid, search):   
    try:
        if search == "grid":
            clf = GridSearchCV(
                estimator=estimator, 
                param_grid=param_grid, 
                scoring=None,
                n_jobs=-1, 
                cv=10, 
                verbose=0,
                return_train_score=True
            )
        elif search == "random":           
            clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=10,
                n_jobs=-1,
                cv=10,
                verbose=0,
                random_state=1,
                return_train_score=True
            )
    except:
        print('Search argument has to be "grid" or "random"')
        sys.exit(0)
        
    # Fit the model
    clf.fit(X=scaled_train, y=y_train)
    
    return clf   

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    classes=["AML", "ALL"]    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.bone)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    thresh = cm.mean()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]), 
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black") 



with open('/student/fengw/ML/project/gene-expression/xtrain.csv') as xtrain:
        x_train=pd.read_csv(xtrain,header=1)
with open('/student/fengw/ML/project/gene-expression/ytrain.csv') as ytrain:
        y_train=pd.read_csv(ytrain,header=1)
with open('/student/fengw/ML/project/gene-expression/xtest.csv') as xtest:
        x_test=pd.read_csv(xtest,header=1)
with open('/student/fengw/ML/project/gene-expression/ytest.csv') as ytest:
        y_test=pd.read_csv(ytest,header=1)

y_train=y_train.replace('ALL',0)
y_train=y_train.replace('AML',1)
y_test=y_test.replace('ALL',0)
y_test=y_test.replace('AML',1)

x_train=x_train[2:].values
y_train=y_train[1:].values
x_test=x_test[2:].values
y_test=y_test[1:].values

knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
probility=knn.predict_proba(x_test)

score=knn.score(x_test,y_test,sample_weight=None)
print 'Accuracy:',score

#knn_param = {
#    "n_neighbors": [i for i in range(1,30,5)],
#    "weights": ["uniform", "distance"],
#    "algorithm": ["ball_tree", "kd_tree", "brute"],
#    "leaf_size": [1, 10, 30],
#    "p": [1,2]
#}

#knn_dist = {
#    "n_neighbors": scipy.stats.randint(1,33),
#    "weights": ["uniform", "distance"],
#    "algorithm": ["ball_tree", "kd_tree", "brute"],
#    "leaf_size": scipy.stats.randint(1,1000),
#    "p": [1,2]
#}


knn_grid = cherchez(KNeighborsClassifier(), knn_param, "grid")
acc = accuracy_score(y_true=y_test, y_pred=knn_grid.predict(scaled_test))
print("**Grid search results**")
print("Best training accuracy:\t", knn_grid.best_score_)
print("Test accuracy:\t", acc)

