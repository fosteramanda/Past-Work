#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program uses Oxford Parkinson's Disease Detection Dataset and utilizes the machine learning models K Nearest Neighbors, Logistic Regression, and
Decision Tree to predict whether or not someone has Parkinson's disease. This dataset is composed of a range of biomedical voice measurements from 31 people,
23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds one of 195 voice recording from these individuals 
("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for 
patients with parkinsons. After performing a straified suffle split to split the data into train and test sets I ran the three models then chose to focus specifically
on Decision Tree. Since the size of tehe training set was small, I trained on all the data but one row then predicted on that row. I then repeated this for each row 
in the dataset. The best accuracy I got with this technique was 0.9179487179487179 which is a good amount greater than the original Decision Tree accuracy which
was 0.7435897435897436. This accuracy was achieved using 194 rows of the 195 row dataset as training data. The ideal model would be achieved if we could train on
all 195 rows and then predict on new data, but since 194 is so similar to 195 we can predict the accuracy of training on 195 would be around that of the 194 model. 
 
@author: amandafoster
"""


# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

### First, pick and prepare a dataset:
parkinsons = pd.read_csv("parkinsons.csv")
parkinsons.info()
parkinsons = parkinsons.drop(["name"], axis=1) #drops the name column
#2: Does a Stratified Shuffle Split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(splitter.split(parkinsons, parkinsons["status"]))
strat_train = parkinsons.iloc[train_index]
strat_test = parkinsons.iloc[test_index]

#3: Split the training and test datasets into inputs and targets dataframes.
strat_train_target = strat_train["status"] #y_train
strat_train_inputs = strat_train.drop(["status"], axis=1) #x_train
    
strat_test_target = strat_test["status"] #y-test
strat_test_inputs = strat_test.drop(["status"], axis=1) #x_test

def runTest(name, model, train_inputs, train_targets, test_inputs, test_targets, avg='weighted', silent=False):
    model.fit(train_inputs, train_targets)
    test_outputs = model.predict(test_inputs)
    score = model.score(test_inputs, test_targets)
    if not silent:
        print("\n" + name + ", Mean test accuracy: ", score)
        print("Precision = TP / (TP + FP) = ", precision_score(test_targets, test_outputs, average=avg))
        print("Recall = TP / (TP + FN) = ", recall_score(test_targets, test_outputs, average=avg))
        print("Confusion Matrix (test):")
        print(confusion_matrix(test_targets, test_outputs))
    return score
    
for (name, model) in [
        ("K Nearest Neighbors classifier", KNeighborsClassifier()), ## looks for similar things. has a concept of distance and looks for patients that are most similar to yours so it makes sense this does best at first  
        ("Logistic Regression", LogisticRegression(solver ="saga")),
        ("Decision Tree", DecisionTreeClassifier(max_depth = 9, min_samples_leaf = 9))]:
    runTest(name,
            model,
            strat_train_inputs,
            strat_train_target,
            strat_test_inputs,
            strat_test_target)
'''
Original Accuracy Values
K Nearest Neighbors, Mean Test Acuracy: 0.8461538461538461
Logistic Regression, Mean test accuracy:  0.7435897435897436
Decision Tree, Mean test accuracy:  0.7435897435897436
'''
#To increase the size of our training set. Larger training set = better model 
def attemptTree(df, max_depth, min_samples_leaf):
    successes = 0
    attempts = 0
    # goes through the entire dataframe n,training on (n-1) data
    for row in range(len(df)):
        reduceddf = df.drop(row) #removes one row from the dataframe
        '''
        reduceddf = reduceddf.drop(row + 1)
        reduceddf = reduceddf.drop(row + 2)
        reduceddf = reduceddf.drop(row + 3)
        reduceddf = reduceddf.drop(row + 4)
        reduceddf = reduceddf.drop(row + 5)
        reduceddf = reduceddf.drop(row + 6)
        reduceddf = reduceddf.drop(row + 7)
        reduceddf = reduceddf.drop(row + 8)
        reduceddf = reduceddf.drop(row + 9)
        '''
        dtc = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf, random_state = 5)
        result = runTest("", dtc,
                         reduceddf.drop(["status"], axis=1), #trains on all the data but the one removed row
                         reduceddf["status"],
                         df[row:row+1].drop(["status"], axis=1), #tests on the one removed row
                         df[row:row+1]["status"],
                         'weighted', True)
        successes += result
        attempts += 1
    #reduceddf.info()
    return successes/attempts

best_max_depth = -1
best_min_samples_leaf = -1
best_score = -1
for max_depth in range(1,11):
    for min_samples_leaf in range(1,11):
        score = attemptTree(parkinsons, max_depth, min_samples_leaf)
        if best_score < score:
            best_score = score
            best_max_depth = max_depth
            best_min_samples_leaf = min_samples_leaf
        print(max_depth, min_samples_leaf, score)

print("Final model: max_depth=",best_max_depth,", min_samples_leaf=",best_min_samples_leaf," score=",best_score)
print("Best Accuracy:" + str(best_score))

'''
What I tested:
    testing on 1: Final model: max_depth= 4 , min_samples_leaf= 7  score= 0.9179487179487179
    testing on 2: Final model: max_depth= 4 , min_samples_leaf= 7  score= 0.8994845360824743
    testing on 3: Final model: max_depth= 4 , min_samples_leaf= 9  score= 0.8687392055267701   
    testing on 4: Final model: max_depth= 5 , min_samples_leaf= 9  score= 0.8372395833333334
    testing on 5: Final model: max_depth= 4 , min_samples_leaf= 1  score= 0.8261780104712044
    testing on 6: Final model: max_depth= 1 , min_samples_leaf= 1  score= 0.8219298245614032
    testing on 7: Final model: max_depth= 1 , min_samples_leaf= 1  score= 0.8231292517006806
    testing on 8: Final model: max_depth= 1 , min_samples_leaf= 1  score= 0.824468085106383
    testing on 9: Final model: max_depth= 1 , min_samples_leaf= 1  score= 0.8259061200237664
    testing on 10: Final model: max_depth= 1 , min_samples_leaf= 1  score= 0.8274193548387108
    
'''

#will be a very similar model as ours above if you trained all 195 inputs vs 194. Then use could use this to predict on new data
bestdtf = DecisionTreeClassifier(max_depth = best_max_depth, min_samples_leaf = best_min_samples_leaf)
bestdtf.fit(parkinsons.drop(["status"], axis=1), parkinsons["status"])  # ideal model

