"""
Created on Tue Dec  3 09:36:32 2019

@author: amandafoster
"""
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import time
import math

def runTest(name, model, train_inputs, train_targets, test_inputs, test_targets, avg='binary'):
    start = time.perf_counter()
    model.fit(train_inputs, train_targets)
    test_outputs = model.predict(test_inputs)
    stop = time.perf_counter()
    elapsed = stop - start
    print("\n" + name + ", Mean test accuracy: ", model.score(test_inputs, test_targets))
    print("Precision = TP / (TP + FP) = ", precision_score(test_targets, test_outputs, average=avg))
    print("Recall = TP / (TP + FN) = ", recall_score(test_targets, test_outputs, average=avg))
    print("Confusion Matrix (test):")
    print(confusion_matrix(test_targets, test_outputs))
    print("The shape after encoding is" + str(train_inputs.shape) + " format (row, column)") #prints out (row, column)
    print('The elapsed time is ' + str(elapsed))
    print("\n")


#1: Loads the dataset and names the columns
mushrooms = pd.read_csv("mushrooms.csv",index_col=False,header=None)
mushrooms.columns = ["Class", "cap-shape", "cap-surface", "cap-color", " bruises", "odor", "gill-attachment", "gill-spacing", 
                  "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                  "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", " ring-number", "ring-type",
                  "spore-print-color", "population", "habitat"]

mushrooms['Class'] = mushrooms['Class'].apply(lambda x: 1 if x=='p' else 0)


#2: Does a Stratified Shuffle Split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(splitter.split(mushrooms, mushrooms["Class"]))
strat_train = mushrooms.iloc[train_index]
strat_test = mushrooms.iloc[test_index]

#3: Split the training and test datasets into inputs and targets dataframes.
strat_train_inputs = strat_train[mushrooms.columns[1:]]
strat_train_target = strat_train[mushrooms.columns[0]]
    
strat_test_inputs = strat_test[mushrooms.columns[1:]]
strat_test_target = strat_test[mushrooms.columns[0]]

class BinaryEncoder():
    def __init__(self):
        self.ordinalMap = {}
        self.columns = []
        self.bits = {}

    def fit(self, X):
        self.ordinalMap = {}
        self.columns = []
        self.bits = {}
        for column in X.columns:
            theMap = {}
            for value in X[column].values:
                if value not in theMap:
                    n = len(theMap)
                    theMap[value] = n
            self.ordinalMap[column] = theMap
            N = len(theMap)
            N -= 1
            bits = 0
            while N>0:
                N = int(N/2)
                self.columns.append(column + str(bits))
                bits = bits + 1
            self.bits[column] = bits
        return self

    def transform(self, X):
        data = []
        for _,rowData in X.iterrows():
            row = []
            for column,value in rowData.items():
                ordinalEncoding = self.ordinalMap[column][value]
                for bit in range(self.bits[column]):
                    row.append(ordinalEncoding % 2)
                    ordinalEncoding = int(ordinalEncoding/2)
            data.append(row)
        return pd.DataFrame(data, columns = self.columns)
    
    #names the encoder
    def __repr__(self):
        return "Binary encoder"

be = BinaryEncoder()
be.fit(mushrooms[mushrooms.columns[1:]])


print("The initial shape of the train data is" + str(strat_train_inputs.shape))

for encoder in [preprocessing.OneHotEncoder(), preprocessing.OrdinalEncoder(), BinaryEncoder()]: #preprocessing.BinaryEncoder()
    
    encoder.fit(mushrooms[mushrooms.columns[1:]]) #fits on all the columns but the class column
    print("Using encoder: ",encoder)
    for (name, model) in [
            ("K Nearest Neighbors classifier", KNeighborsClassifier()),
            ("Logistic Regression", LogisticRegression(solver ="saga")),
            ("Decision Tree", DecisionTreeClassifier(max_depth = 9, min_samples_leaf = 9))]:
        runTest(name,
                model,
                encoder.transform(strat_train_inputs),
                strat_train_target,
                encoder.transform(strat_test_inputs),
                strat_test_target)
    
    #print(ohe.transform(strat_train_inputs).shape)
    
#runBinary("K Nearest Neighbors classifier", KNeighborsClassifier(), strat_train_target, strat_test_target)
    
