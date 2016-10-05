# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 08:03:58 2016

@author: Kushal
"""

from class_vis import prettyPicture
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import make_scorer
from titanic_visualizations import survival_stats
from sklearn.metrics import f1_score
student_data=pd.read_csv("student-data.csv")

# TODO: Calculate number of students
n_students = len(student_data)

# TODO: Calculate number of features
n_features = len(student_data.columns)

# TODO: Calculate passing students
n_passed = 0;
for i in student_data['passed']:
    if i == 'yes':
        n_passed+=1

# TODO: Calculate failing students
n_failed = n_students-n_passed

# TODO: Calculate graduation rate
grad_rate = (n_passed*100.0)/n_students

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

X_all = student_data[feature_cols]
y_all = student_data[target_col]

# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split as tts
# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train = None
X_test = None
y_train = None
y_test = None
X_train,X_test,y_train,y_test=tts(X_all,y_all,test_size=(num_test)/395.0,random_state=1)
# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])






from sklearn.cross_validation import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier
clf1=DecisionTreeClassifier()

from sklearn.svm import SVC
clf2=SVC(kernel='rbf')
#try all kernels

from sklearn.neighbors import KNeighborsClassifier
clf3=KNeighborsClassifier(n_neighbors=4)

from sklearn.naive_bayes import GaussianNB
clf4=GaussianNB()

from sklearn.ensemble import BaggingClassifier
clf5=BaggingClassifier()

from sklearn.ensemble import ExtraTreesClassifier
clf6=ExtraTreesClassifier()

from sklearn.ensemble import RandomForestClassifier
clf7=RandomForestClassifier()

from sklearn.linear_model import SGDClassifier
clf8=SGDClassifier()

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 09:41:39 2016

@author: Kushal
"""

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    

print "\n\n\nclassifier1:"
train_predict(clf1, X_train, y_train, X_test, y_test)

print "\n\n\nclassifier2:"
train_predict(clf2, X_train, y_train, X_test, y_test)
print "\n\n\nclassifier3:"
train_predict(clf3, X_train, y_train, X_test, y_test)
print "\n\n\nclassifier4:"
train_predict(clf4, X_train, y_train, X_test, y_test)
print "\n\n\nclassifier5:"
train_predict(clf5, X_train, y_train, X_test, y_test)
print "\n\n\nclassifier6:"
train_predict(clf6, X_train, y_train, X_test, y_test)
print "\n\n\nclassifier7:"
train_predict(clf7, X_train, y_train, X_test, y_test)
print "\n\n\nclassifier8:"
train_predict(clf8, X_train, y_train, X_test, y_test)


survival_stats(X_all,y_all,'paid')